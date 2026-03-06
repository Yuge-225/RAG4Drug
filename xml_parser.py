"""
xml_parser.py
─────────────────────────────────────────────────────────────
从 drug_db.xml 解析数据，同时写入：
  1. SQLite  → 结构化数据（药物信息、相互作用、食物禁忌等）
  2. ChromaDB → 向量化文本（机制、描述、适应症等）

特性：
  - iterparse 流式解析，2GB文件不崩
  - Rich 终端可视化，实时显示进度
  - 断点续传，中途崩了从上次继续
  - 批量embedding，减少API调用次数

依赖安装：
  pip install rich langchain-chroma langchain-openai langchain-text-splitters
"""

import xml.etree.ElementTree as ET
import sqlite3
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.rule import Rule

# ── 你现有的配置（直接复用）─────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 常量配置 ────────────────────────────────────────────────
XML_PATH         = "./drug_db.xml"
SQLITE_PATH      = "./drug_structured.db"
CHROMA_DIR       = "./chroma_db"
COLLECTION_NAME  = "drug_rag"
EMBED_BATCH_SIZE = 50      # 每批embedding的药物数量
MAX_DRUGS        = None    # None = 全量；调试时设成 100

NS = {'db': 'http://www.drugbank.ca'}

console = Console()


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def safe_text(element, path):
    """安全获取XML节点文本，找不到返回空字符串"""
    node = element.find(path, NS)
    return node.text.strip() if node is not None and node.text else ""


def extract_cyp(text: str) -> list:
    """从文本中提取CYP450酶名称"""
    return list(set(re.findall(r'CYP\s*\d+[A-Z]\d*', text.upper()))) if text else []


def extract_severity(description: str) -> str:
    d = description.lower()
    
    if any(w in d for w in [
        'severe', 'fatal', 'life-threatening',
        'contraindicated', 'avoid', 'do not use',
        'serious risk', 'significantly increase'
    ]):
        return 'severe'
    
    if any(w in d for w in [
        'moderate',
        'risk or severity',          # ← DrugBank常用句式
        'may increase the risk',     # ← DrugBank常用句式
        'can be increased',          # ← DrugBank常用句式
        'caution', 'monitor',
        'may increase', 'may decrease',
        'increase the risk'          # ← DrugBank常用句式
    ]):
        return 'moderate'
    
    if any(w in d for w in [
        'minor', 'mild', 'slight', 'small', 'minimal'
    ]):
        return 'minor'
    
    return 'unknown'

def chunk_text(text: str, max_len: int = 800) -> list:
    """按句子切块，避免单个chunk过长"""
    if len(text) <= max_len:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) > max_len and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += (" " if current else "") + sent
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text[:max_len]]


# ══════════════════════════════════════════════════════════════
# SQLite：建表 + 写入
# ══════════════════════════════════════════════════════════════

def init_sqlite(db_path: str) -> sqlite3.Connection:
    """初始化SQLite，创建所有表"""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        -- 药物基本信息
        CREATE TABLE IF NOT EXISTS drugs (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            state       TEXT,
            half_life   TEXT,
            groups      TEXT,
            cas_number  TEXT,
            description TEXT
        );

        -- 药物别名（NER查询用）
        CREATE TABLE IF NOT EXISTS synonyms (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_id  TEXT NOT NULL,
            synonym  TEXT NOT NULL
        );

        -- 药物相互作用（核心表）
        CREATE TABLE IF NOT EXISTS drug_interactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_id_a   TEXT NOT NULL,
            drug_id_b   TEXT NOT NULL,
            severity    TEXT,
            description TEXT
        );

        -- 食物相互作用
        CREATE TABLE IF NOT EXISTS food_interactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_id     TEXT NOT NULL,
            description TEXT
        );

        -- 剂量信息
        CREATE TABLE IF NOT EXISTS dosages (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_id  TEXT NOT NULL,
            form     TEXT,
            route    TEXT,
            strength TEXT
        );

        -- CYP酶信息（后续知识图谱用）
        CREATE TABLE IF NOT EXISTS enzymes (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_id  TEXT NOT NULL,
            cyp_name TEXT NOT NULL
        );

        -- 断点续传进度表
        CREATE TABLE IF NOT EXISTS parse_progress (
            drug_id     TEXT PRIMARY KEY,
            status      TEXT,         -- 'done' / 'failed'
            finished_at TEXT
        );

        -- 索引加速查询
        CREATE INDEX IF NOT EXISTS idx_synonyms_drug    ON synonyms(drug_id);
        CREATE INDEX IF NOT EXISTS idx_synonyms_name    ON synonyms(synonym COLLATE NOCASE);
        CREATE INDEX IF NOT EXISTS idx_interactions_a   ON drug_interactions(drug_id_a);
        CREATE INDEX IF NOT EXISTS idx_interactions_b   ON drug_interactions(drug_id_b);
        CREATE INDEX IF NOT EXISTS idx_food_drug        ON food_interactions(drug_id);
        CREATE INDEX IF NOT EXISTS idx_dosages_drug     ON dosages(drug_id);
        CREATE INDEX IF NOT EXISTS idx_enzymes_drug     ON enzymes(drug_id);
    """)
    conn.commit()
    return conn


def get_processed_ids(conn: sqlite3.Connection) -> set:
    """读取已处理完成的drug_id（断点续传）"""
    rows = conn.execute(
        "SELECT drug_id FROM parse_progress WHERE status = 'done'"
    ).fetchall()
    return {r[0] for r in rows}


def write_drug_to_sqlite(conn: sqlite3.Connection, drug_data: dict):
    """将一个药物的所有结构化数据写入SQLite"""
    cur = conn.cursor()
    d = drug_data

    # drugs 主表
    cur.execute("""
        INSERT OR REPLACE INTO drugs
            (id, name, state, half_life, groups, cas_number, description)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (d['id'], d['name'], d['state'], d['half_life'],
          d['groups'], d['cas_number'], d['description'][:500] if d['description'] else ''))

    # synonyms（包含name本身）
    all_synonyms = list(set([d['name']] + d['synonyms']))
    for syn in all_synonyms:
        if syn:
            cur.execute(
                "INSERT INTO synonyms (drug_id, synonym) VALUES (?, ?)",
                (d['id'], syn)
            )

    # drug_interactions
    for ix in d['interactions']:
        cur.execute("""
            INSERT INTO drug_interactions
                (drug_id_a, drug_id_b, severity, description)
            VALUES (?, ?, ?, ?)
        """, (d['id'], ix['target_id'], ix['severity'], ix['description'][:600]))

    # food_interactions
    for fi in d['food_interactions']:
        cur.execute(
            "INSERT INTO food_interactions (drug_id, description) VALUES (?, ?)",
            (d['id'], fi)
        )

    # dosages
    for ds in d['dosages']:
        cur.execute("""
            INSERT INTO dosages (drug_id, form, route, strength)
            VALUES (?, ?, ?, ?)
        """, (d['id'], ds['form'], ds['route'], ds['strength']))

    # enzymes（CYP）
    for cyp in d['cyp_enzymes']:
        cur.execute(
            "INSERT INTO enzymes (drug_id, cyp_name) VALUES (?, ?)",
            (d['id'], cyp)
        )

    # 标记完成
    cur.execute("""
        INSERT OR REPLACE INTO parse_progress (drug_id, status, finished_at)
        VALUES (?, 'done', ?)
    """, (d['id'], datetime.now().isoformat()))

    conn.commit()


# ══════════════════════════════════════════════════════════════
# XML 解析：提取单个 drug 节点
# ══════════════════════════════════════════════════════════════

def parse_drug_element(elem) -> dict:
    """从一个 <drug> XML元素提取所有需要的字段"""

    drug_id   = safe_text(elem, 'db:drugbank-id')
    name      = safe_text(elem, 'db:name')
    desc      = safe_text(elem, 'db:description')
    mechanism = safe_text(elem, 'db:mechanism-of-action')
    pharma    = safe_text(elem, 'db:pharmacodynamics')
    indication= safe_text(elem, 'db:indication')
    metabolism= safe_text(elem, 'db:metabolism')
    state     = safe_text(elem, 'db:state')
    half_life = safe_text(elem, 'db:half-life')
    cas       = safe_text(elem, 'db:cas-number')
    toxicity  = safe_text(elem, 'db:toxicity')

    groups = [
        g.text for g in elem.findall('db:groups/db:group', NS) if g.text
    ]

    synonyms = [
        s.text.strip()
        for s in elem.findall('db:synonyms/db:synonym', NS)
        if s.text
    ]

    # 相互作用
    interactions = []
    for ix in elem.findall('db:drug-interactions/db:drug-interaction', NS):
        target_id = safe_text(ix, 'db:drugbank-id')
        ix_desc   = safe_text(ix, 'db:description')
        if target_id:
            interactions.append({
                'target_id':   target_id,
                'severity':    extract_severity(ix_desc),
                'description': ix_desc
            })

    # 食物相互作用
    food_interactions = [
        fi.text.strip()
        for fi in elem.findall('db:food-interactions/db:food-interaction', NS)
        if fi.text
    ]

    # 剂量
    dosages = []
    for ds in elem.findall('db:dosages/db:dosage', NS):
        dosages.append({
            'form':     safe_text(ds, 'db:form'),
            'route':    safe_text(ds, 'db:route'),
            'strength': safe_text(ds, 'db:strength'),
        })

    # CYP酶（从metabolism和mechanism提取）
    cyp_text = metabolism + " " + mechanism
    cyp_enzymes = extract_cyp(cyp_text)

    # 向量化文本（只索引有实质内容的字段）
    texts_for_embedding = []
    fields_map = {
        'description':  desc,
        'mechanism':    mechanism,
        'indication':   indication,
        'pharmacodynamics': pharma,
    }
    for field_name, text in fields_map.items():
        if text and len(text) > 80:  # 太短的没有意义
            for i, chunk in enumerate(chunk_text(text)):
                texts_for_embedding.append({
                    'chunk_id': f"{drug_id}_{field_name}_{i}",
                    'text':     chunk,
                    'metadata': {
                        'drug_id':   drug_id,
                        'drug_name': name,
                        'field':     field_name,
                    }
                })

    return {
        'id':               drug_id,
        'name':             name,
        'description':      desc,
        'state':            state,
        'half_life':        half_life,
        'groups':           ','.join(groups),
        'cas_number':       cas,
        'synonyms':         synonyms,
        'interactions':     interactions,
        'food_interactions':food_interactions,
        'dosages':          dosages,
        'cyp_enzymes':      cyp_enzymes,
        'texts_for_embedding': texts_for_embedding,
    }


# ══════════════════════════════════════════════════════════════
# ChromaDB：批量写入
# ══════════════════════════════════════════════════════════════

def init_chroma(chroma_dir: str, collection_name: str):
    """初始化ChromaDB，复用你现有的embedding模型"""
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    chroma = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=chroma_dir,
    )
    return chroma


def flush_embed_batch(chroma, batch: list, stats: dict):
    """把积累的文本块批量写入ChromaDB"""
    if not batch:
        return
    ids   = [item['chunk_id'] for item in batch]
    texts = [item['text']     for item in batch]
    metas = [item['metadata'] for item in batch]
    try:
        chroma.add_texts(texts=texts, metadatas=metas, ids=ids)
        stats['chroma_chunks'] += len(batch)
    except Exception as e:
        stats['chroma_errors'] += 1
        console.log(f"[red]ChromaDB写入错误: {e}[/red]")


# ══════════════════════════════════════════════════════════════
# Rich 可视化：构建实时面板
# ══════════════════════════════════════════════════════════════

def make_stats_panel(stats: dict, start_time: float) -> Panel:
    """构建右侧统计面板"""
    elapsed = time.time() - start_time
    speed = stats['processed'] / elapsed if elapsed > 0 else 0

    # 预估剩余时间
    if stats['total'] > 0 and speed > 0:
        remaining = (stats['total'] - stats['processed']) / speed
        eta = str(timedelta(seconds=int(remaining)))
    else:
        eta = "计算中..."

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("指标", style="dim", width=16)
    table.add_column("数值", style="bold")

    table.add_row("处理速度",    f"[cyan]{speed:.1f}[/cyan] 药物/秒")
    table.add_row("预计完成",    f"[yellow]{eta}[/yellow]")
    table.add_row("",            "")
    table.add_row("✅ 已完成",   f"[green]{stats['processed']:,}[/green]")
    table.add_row("⏭️  跳过",    f"[dim]{stats['skipped']:,}[/dim]")
    table.add_row("❌ 错误",     f"[red]{stats['errors']:,}[/red]")
    table.add_row("",            "")
    table.add_row("📊 SQLite行", f"[blue]{stats['sql_rows']:,}[/blue]")
    table.add_row("  相互作用",  f"[magenta]{stats['interactions']:,}[/magenta]")
    table.add_row("  同义词",    f"[magenta]{stats['synonyms']:,}[/magenta]")
    table.add_row("",            "")
    table.add_row("🧠 向量块",   f"[blue]{stats['chroma_chunks']:,}[/blue]")
    table.add_row("  待写入",    f"[yellow]{stats['embed_pending']:,}[/yellow]")
    if stats['chroma_errors'] > 0:
        table.add_row("  写入错误",f"[red]{stats['chroma_errors']:,}[/red]")

    return Panel(table, title="[bold]实时统计[/bold]", border_style="blue")


def make_current_drug_panel(drug_data: dict) -> Panel:
    """构建当前正在处理的药物信息面板"""
    if not drug_data:
        return Panel("[dim]等待中...[/dim]", title="当前处理", border_style="dim")

    content = Text()
    content.append(f"  ID:    ", style="dim")
    content.append(f"{drug_data.get('id', '')}\n", style="cyan bold")
    content.append(f"  名称:  ", style="dim")
    content.append(f"{drug_data.get('name', '')}\n", style="white bold")
    content.append(f"  相互作用: ", style="dim")
    content.append(f"{len(drug_data.get('interactions', []))} 条\n", style="magenta")
    content.append(f"  向量块:   ", style="dim")
    content.append(f"{len(drug_data.get('texts_for_embedding', []))} 块\n", style="green")
    content.append(f"  CYP酶:    ", style="dim")
    cyps = drug_data.get('cyp_enzymes', [])
    content.append(f"{', '.join(cyps) if cyps else '无'}\n", style="yellow")

    return Panel(content, title="[bold]当前处理[/bold]", border_style="green")


def make_severity_panel(stats: dict) -> Panel:
    """相互作用严重程度分布"""
    total_ix = stats['interactions']
    if total_ix == 0:
        return Panel("[dim]暂无数据[/dim]", title="严重程度分布", border_style="dim")

    sev = stats['severity']
    table = Table(box=box.SIMPLE, show_header=False, padding=(0,1))
    table.add_column("等级", width=10)
    table.add_column("数量", width=8)
    table.add_column("占比", width=20)

    for level, color, label in [
        ('severe',   'red',    '🔴 严重'),
        ('moderate', 'yellow', '🟡 中度'),
        ('minor',    'green',  '🟢 轻微'),
        ('unknown',  'dim',    '⚪ 未知'),
    ]:
        count = sev.get(level, 0)
        pct   = count / total_ix if total_ix > 0 else 0
        bar   = "█" * int(pct * 15) + "░" * (15 - int(pct * 15))
        table.add_row(
            f"[{color}]{label}[/{color}]",
            f"[{color}]{count:,}[/{color}]",
            f"[{color}]{bar}[/{color}] {pct:.1%}"
        )

    return Panel(table, title="[bold]相互作用严重程度[/bold]", border_style="magenta")


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def count_drugs(xml_path: str) -> int:
    count = 0
    for event, elem in ET.iterparse(xml_path, events=['end']):
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'drug':
            drug_id = safe_text(elem, 'db:drugbank-id')
            if drug_id:            # ← 只有顶层drug才有drugbank-id
                count += 1
            elem.clear()
    return count

def run_parser(
    xml_path:    str  = XML_PATH,
    sqlite_path: str  = SQLITE_PATH,
    chroma_dir:  str  = CHROMA_DIR,
    max_drugs:   int  = MAX_DRUGS,
    batch_size:  int  = EMBED_BATCH_SIZE,
):
    start_time = time.time()

    # ── 启动画面 ─────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold cyan]DrugBank XML Parser[/bold cyan]\n"
        "[dim]SQLite + ChromaDB 双路写入[/dim]",
        border_style="cyan"
    ))
    console.print()

    # ── 检查文件 ─────────────────────────────────────────────
    if not Path(xml_path).exists():
        console.print(f"[red]❌ 找不到文件: {xml_path}[/red]")
        return

    file_size_gb = Path(xml_path).stat().st_size / (1024**3)
    console.print(f"📄 XML文件: [cyan]{xml_path}[/cyan] "
                  f"([yellow]{file_size_gb:.2f} GB[/yellow])")

    # ── 统计总药物数 ─────────────────────────────────────────
    total_drugs = count_drugs(xml_path)
    if max_drugs:
        total_drugs = min(total_drugs, max_drugs)
    console.print(f"💊 药物总数: [cyan]{total_drugs:,}[/cyan]")
    console.print()

    # ── 初始化数据库 ─────────────────────────────────────────
    console.print("🔧 初始化 SQLite...")
    conn = init_sqlite(sqlite_path)
    processed_ids = get_processed_ids(conn)
    console.print(f"   断点续传：已有 [green]{len(processed_ids):,}[/green] 个药物处理完成")

    console.print("🔧 初始化 ChromaDB...")
    try:
        chroma = init_chroma(chroma_dir, COLLECTION_NAME)
        use_chroma = True
        console.print("   ChromaDB 连接成功 ✅")
    except Exception as e:
        console.print(f"   [yellow]⚠️  ChromaDB初始化失败: {e}[/yellow]")
        console.print("   [dim]将只写入SQLite，跳过向量化[/dim]")
        chroma = None
        use_chroma = False

    console.print()
    console.print(Rule("[dim]开始解析[/dim]"))
    console.print()

    # ── 统计数据 ─────────────────────────────────────────────
    stats = {
        'total':        total_drugs,
        'processed':    0,
        'skipped':      len(processed_ids),
        'errors':       0,
        'sql_rows':     0,
        'interactions': 0,
        'synonyms':     0,
        'chroma_chunks':0,
        'chroma_errors':0,
        'embed_pending':0,
        'severity':     {'severe':0,'moderate':0,'minor':0,'unknown':0},
    }

    embed_batch  = []   # 积累待embedding的文本块
    current_drug = {}   # 当前正在处理的药物（显示用）
    drug_count   = 0    # 包含跳过的总计数

    # ── Rich Live 布局 ────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:

        task = progress.add_task(
            "解析药物数据...",
            total=total_drugs - len(processed_ids)
        )

        # ── iterparse 流式扫描 ────────────────────────────────
        for event, elem in ET.iterparse(xml_path, events=['end']):
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if tag != 'drug':
                continue

            if max_drugs and drug_count >= max_drugs:
                elem.clear()
                break

            drug_count += 1

            # ── 解析字段 ──────────────────────────────────────
            try:
                drug_id = safe_text(elem, 'db:drugbank-id')
                if not drug_id:
                    elem.clear()
                    continue

                # 断点续传：已处理过的跳过
                if drug_id in processed_ids:
                    elem.clear()
                    continue

                drug_data = parse_drug_element(elem)
                current_drug = drug_data

                # ── 写SQLite ──────────────────────────────────
                write_drug_to_sqlite(conn, drug_data)

                # 更新统计
                stats['interactions'] += len(drug_data['interactions'])
                stats['synonyms']     += len(drug_data['synonyms'])
                stats['sql_rows']     += (
                    1 +
                    len(drug_data['synonyms']) +
                    len(drug_data['interactions']) +
                    len(drug_data['food_interactions']) +
                    len(drug_data['dosages']) +
                    len(drug_data['cyp_enzymes'])
                )
                for ix in drug_data['interactions']:
                    sev = ix.get('severity', 'unknown')
                    stats['severity'][sev] = stats['severity'].get(sev, 0) + 1

                # ── 积累embedding batch ───────────────────────
                if use_chroma:
                    embed_batch.extend(drug_data['texts_for_embedding'])
                    stats['embed_pending'] = len(embed_batch)

                    # 达到批量阈值 → 写ChromaDB
                    if len(embed_batch) >= batch_size * 4:
                        flush_embed_batch(chroma, embed_batch, stats)
                        embed_batch = []
                        stats['embed_pending'] = 0

                stats['processed'] += 1
                progress.advance(task)

            except Exception as e:
                stats['errors'] += 1
                console.log(f"[red]解析错误 {drug_id}: {e}[/red]")

            finally:
                elem.clear()  # 关键：释放内存

        # ── 处理剩余batch ─────────────────────────────────────
        if use_chroma and embed_batch:
            progress.update(task, description="写入剩余向量块...")
            flush_embed_batch(chroma, embed_batch, stats)
            stats['embed_pending'] = 0

    # ── 完成报告 ─────────────────────────────────────────────
    elapsed = time.time() - start_time
    console.print()
    console.print(Rule("[green]✅ 解析完成[/green]"))
    console.print()

    # 最终统计表
    final_table = Table(
        title="📊 最终统计报告",
        box=box.ROUNDED,
        border_style="green",
        show_header=True,
        header_style="bold green"
    )
    final_table.add_column("类别",   style="dim",   width=20)
    final_table.add_column("数量",   style="cyan",  width=15, justify="right")
    final_table.add_column("说明",   style="white", width=30)

    final_table.add_row("处理药物数",     f"{stats['processed']:,}",    "成功写入SQLite")
    final_table.add_row("跳过（续传）",   f"{stats['skipped']:,}",     "已在上次处理")
    final_table.add_row("错误数",         f"{stats['errors']:,}",      "解析失败")
    final_table.add_row("─"*18,           "─"*13,                      "─"*28)
    final_table.add_row("SQLite总行数",   f"{stats['sql_rows']:,}",    "所有表合计")
    final_table.add_row("  相互作用记录", f"{stats['interactions']:,}","drug_interactions表")
    final_table.add_row("  同义词记录",   f"{stats['synonyms']:,}",    "synonyms表（NER用）")
    final_table.add_row("─"*18,           "─"*13,                      "─"*28)
    final_table.add_row("ChromaDB向量块", f"{stats['chroma_chunks']:,}","已写入向量库")
    final_table.add_row("─"*18,           "─"*13,                      "─"*28)
    final_table.add_row("总耗时",
                         str(timedelta(seconds=int(elapsed))),
                         f"约 {stats['processed']/elapsed:.1f} 药物/秒")

    console.print(final_table)
    console.print()

    # 严重程度分布
    sev_table = Table(title="相互作用严重程度分布", box=box.SIMPLE)
    sev_table.add_column("等级")
    sev_table.add_column("数量", justify="right")
    sev_table.add_column("占比", justify="right")
    total_ix = stats['interactions']
    for level, color in [('severe','red'),('moderate','yellow'),
                          ('minor','green'),('unknown','dim')]:
        cnt = stats['severity'].get(level, 0)
        pct = f"{cnt/total_ix:.1%}" if total_ix > 0 else "0%"
        sev_table.add_row(
            f"[{color}]{level}[/{color}]",
            f"[{color}]{cnt:,}[/{color}]",
            f"[{color}]{pct}[/{color}]"
        )
    console.print(sev_table)
    console.print()

    # 输出文件位置
    console.print(Panel(
        f"[bold]SQLite:[/bold]  [cyan]{sqlite_path}[/cyan]\n"
        f"[bold]ChromaDB:[/bold] [cyan]{chroma_dir}/[/cyan]\n\n"
        f"[dim]下一步：修改 rag.py 接入双路召回[/dim]",
        title="📁 输出文件",
        border_style="blue"
    ))

    conn.close()


# ══════════════════════════════════════════════════════════════
# 快速验证：解析完成后抽查数据质量
# ══════════════════════════════════════════════════════════════

def verify_results(sqlite_path: str = SQLITE_PATH):
    """解析完成后，抽查几条数据验证质量"""
    console.print()
    console.print(Rule("[cyan]数据质量抽查[/cyan]"))

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    checks = [
        ("drugs表总数",           "SELECT COUNT(*) as n FROM drugs"),
        ("synonyms表总数",        "SELECT COUNT(*) as n FROM synonyms"),
        ("drug_interactions总数", "SELECT COUNT(*) as n FROM drug_interactions"),
        ("SEVERE相互作用数",      "SELECT COUNT(*) as n FROM drug_interactions WHERE severity='severe'"),
        ("food_interactions总数", "SELECT COUNT(*) as n FROM food_interactions"),
    ]

    table = Table(box=box.SIMPLE)
    table.add_column("检查项",  style="dim")
    table.add_column("结果",    style="cyan", justify="right")

    for label, sql in checks:
        row = conn.execute(sql).fetchone()
        table.add_row(label, f"{row['n']:,}")

    console.print(table)

    # 随机抽查一条相互作用
    console.print("\n[bold]随机抽查一条相互作用：[/bold]")
    row = conn.execute("""
        SELECT a.name as drug_a, b.name as drug_b,
               di.severity, di.description
        FROM drug_interactions di
        JOIN drugs a ON a.id = di.drug_id_a
        JOIN drugs b ON b.id = di.drug_id_b
        WHERE di.severity = 'severe'
        ORDER BY RANDOM() LIMIT 1
    """).fetchone()

    if row:
        console.print(f"  [red]{row['drug_a']}[/red] × [red]{row['drug_b']}[/red]")
        console.print(f"  严重程度: [bold red]{row['severity']}[/bold red]")
        console.print(f"  描述: [dim]{row['description'][:150]}...[/dim]")

    # NER测试
    console.print("\n[bold]NER同义词测试（搜索 'warfarin'）：[/bold]")
    rows = conn.execute("""
        SELECT d.name, s.synonym
        FROM synonyms s JOIN drugs d ON d.id = s.drug_id
        WHERE LOWER(s.synonym) LIKE '%warfarin%'
        LIMIT 5
    """).fetchall()
    for r in rows:
        console.print(f"  [cyan]{r['synonym']}[/cyan] → {r['name']}")

    conn.close()


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DrugBank XML Parser')
    parser.add_argument('--xml',      default=XML_PATH,    help='XML文件路径')
    parser.add_argument('--sqlite',   default=SQLITE_PATH, help='SQLite输出路径')
    parser.add_argument('--chroma',   default=CHROMA_DIR,  help='ChromaDB目录')
    parser.add_argument('--max',      type=int, default=None, help='最多处理N个药物（调试用）')
    parser.add_argument('--verify',   action='store_true',   help='只做数据质量抽查')
    args = parser.parse_args()

    if args.verify:
        verify_results(args.sqlite)
    else:
        run_parser(
            xml_path    = args.xml,
            sqlite_path = args.sqlite,
            chroma_dir  = args.chroma,
            max_drugs   = args.max,
        )
        verify_results(args.sqlite)
