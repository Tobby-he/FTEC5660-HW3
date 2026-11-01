#!/usr/bin/env python3
"""FTEC5660 HW3 student starter: an agent that applies 3 of Fisher's 15 points using tools."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sqlite3
import threading
import zlib
from pathlib import Path
from typing import Any


MODEL_NAME = "deepseek-v4-flash"
INTERVIEW_LIMIT = 10
MAX_SQL_ROWS = 50

# Philip Fisher, "Common Stocks and Uncommon Profits", Part One, Chapter 3.
# For every point, PASS means the company looks good on it. You choose 3
# points and decide yourself what PASS means for each of them.
FISHER_POINTS = {
    1: "Does the company have products or services with sufficient market potential to make possible a sizable increase in sales for at least several years?",
    2: "Does the management have a determination to continue to develop products or processes that will still further increase total sales potentials when the growth potentials of currently attractive product lines have largely been exploited?",
    3: "How effective are the company's research and development efforts in relation to its size?",
    4: "Does the company have an above-average sales organization?",
    5: "Does the company have a worthwhile profit margin?",
    6: "What is the company doing to maintain or improve profit margins?",
    7: "Does the company have outstanding labor and personnel relations?",
    8: "Does the company have outstanding executive relations?",
    9: "Does the company have depth to its management?",
    10: "How good are the company's cost analysis and accounting controls?",
    11: "Are there other aspects of the business, somewhat peculiar to the industry involved, which will give the investor important clues as to how outstanding the company may be in relation to its competition?",
    12: "Does the company have a short-range or long-range outlook in regard to profits?",
    13: "In the foreseeable future will the growth of the company require sufficient equity financing so that the larger number of shares then outstanding will largely cancel the existing stockholders' benefit from this anticipated growth?",
    14: "Does the management talk freely to investors about its affairs when things are going well but 'clam up' when troubles and disappointments occur?",
    15: "Does the company have a management of unquestionable integrity?",
}
VERDICTS = ("PASS", "FAIL")


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# Choose any 3 of the 15 points, e.g. CHOSEN_POINTS = [1, 7, 15]. Your agent
# gives a PASS or FAIL verdict on these 3 points only, with your own rule for each.
CHOSEN_POINTS: list[int] = [4, 7, 11]  ### YOUR CODE HERE

from typing import Literal  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

# Every chosen point is a "scuttlebutt" point: the evidence is what outsiders
# and staff say in interviews. For each point we route to its topic and to the
# two roles that know about it (README, "Who knows about it"), and say in plain
# words what favourable and unfavourable evidence looks like.
POINT_PLANS: dict[int, dict[str, Any]] = {
    4: {
        "name": "Above-average sales organization",
        "topic": "sales_team",
        "roles": ["customer", "competitor"],
        "favourable": "the sales force is skilled, proactive and consultative: reps know the "
                      "customer's products, visit, solve problems, are better trained than rivals' "
                      "or win head-to-head deals",
        "unfavourable": "the sales force is passive, weak or unknowledgeable: reps only send price "
                        "lists, keep changing, do not know the product, are not a factor in deals "
                        "or lose customers to rivals",
    },
    7: {
        "name": "Outstanding labour and personnel relations",
        "topic": "working_conditions",
        "roles": ["current_employee", "former_employee"],
        "favourable": "staff are treated well: fair pay, good training, low turnover, people stay "
                      "for years, problems get fixed (a former employee who left for a personal "
                      "reason such as moving city but praises the company is favourable)",
        "unfavourable": "relations are poor: high turnover, constant overtime, people quitting, low "
                        "morale, strikes or strike threats, complaints about pay",
    },
    11: {
        "name": "Advantage over competitors",
        "topic": "competitive_edge",
        "roles": ["competitor", "customer"],
        "favourable": "the company has a durable edge that rivals cannot easily match: high "
                      "switching costs, know-how or installed base that is hard to copy, "
                      "unmatched performance, customers who leave come back",
        "unfavourable": "the company has no real edge: its products are a commodity, "
                        "interchangeable with other suppliers, easy to copy, or bought purely on price",
    },
}

# If nobody who knows gives a usable answer, fall back to the more common verdict.
DEFAULT_VERDICT = "PASS"
MAX_PARALLEL_COMPANIES = 8
MAX_ATTEMPTS = 3


class WitnessStance(BaseModel):
    """How one interview answer bears on a Fisher point."""

    stance: Literal["favourable", "unfavourable", "no_comment"] = Field(
        description="favourable or unfavourable for the company on this point; no_comment if the "
                    "person says they cannot comment or the answer says nothing about the point"
    )
    reason: str = Field(description="One short sentence citing the words that decided the stance.")


class ReflectedVerdict(BaseModel):
    """A final verdict after re-reading conflicting or unclear evidence."""

    verdict: Literal["PASS", "FAIL"]
    reason: str = Field(description="One short sentence explaining the verdict.")


STANCE_PROMPT = """You are an equity analyst applying Philip Fisher's "scuttlebutt" method.
Fisher's point {point}: {question}

Evidence counts as FAVOURABLE when {favourable}.
Evidence counts as UNFAVOURABLE when {unfavourable}.
If the person says they cannot comment, or says nothing relevant, the stance is no_comment.

A {role} was asked about the company's {topic}. Their answer:
{answer}

Classify this single answer."""

REFLECT_PROMPT = """You are checking an analyst's work on Philip Fisher's point {point}: {question}

PASS means the company looks favourable on this point. It is favourable when {favourable}.
It is unfavourable when {unfavourable}.

The interview evidence and the first reading of each answer:
{evidence}

The readings conflict or are unclear. Re-read the quoted answers themselves (not the readings),
weigh them, and give the final PASS or FAIL verdict."""

SUMMARY_PROMPT = """Write a short summary (2 to 3 sentences, plain prose, no lists) of this Fisher
analysis of {company}. Explain each verdict with the interview evidence behind it; quote a few
words where it helps. Do not invent facts.

{findings}"""


def build_agent() -> Any:
    """Create the DeepSeek model and the three prompt chains used for every company.

    The tools differ per company, so the routing and the interviews happen in
    ``analyze_companies``. Here we build what is shared: a structured
    classifier for single interview answers, a reflection step for points
    whose evidence conflicts, and a summary writer.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    # Thinking mode rejects the forced tool call that structured output uses,
    # and these one-answer classifications do not need it.
    model = ChatDeepSeek(
        model=MODEL_NAME, temperature=0, max_retries=3, timeout=120,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return {
        "stance": ChatPromptTemplate.from_template(STANCE_PROMPT)
        | model.with_structured_output(WitnessStance),
        "reflect": ChatPromptTemplate.from_template(REFLECT_PROMPT)
        | model.with_structured_output(ReflectedVerdict),
        "summary": ChatPromptTemplate.from_template(SUMMARY_PROMPT) | model,
    }


async def _with_retries(coro_factory: Any) -> Any:
    """Await ``coro_factory()``, retrying with a short backoff on any error."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return await coro_factory()
        except Exception:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            await asyncio.sleep(2 * (attempt + 1))


def _point_inputs(point: int) -> dict[str, str]:
    plan = POINT_PLANS[point]
    return {
        "point": str(point),
        "question": FISHER_POINTS[point],
        "favourable": plan["favourable"],
        "unfavourable": plan["unfavourable"],
    }


async def _judge_point(agent: dict[str, Any], interview: Any, point: int) -> dict[str, Any]:
    """Interview the people who know, classify each answer, and reflect if they disagree."""
    plan = POINT_PLANS[point]
    topic = plan["topic"]
    answers = await asyncio.gather(*(
        asyncio.to_thread(interview.invoke, {"role": role, "topic": topic}) for role in plan["roles"]
    ))
    stances = await asyncio.gather(*(
        _with_retries(lambda role=role, answer=answer: agent["stance"].ainvoke(
            {**_point_inputs(point), "role": role.replace("_", " "), "topic": topic.replace("_", " "),
             "answer": answer}
        ))
        for role, answer in zip(plan["roles"], answers)
    ))
    witnesses = [
        {"role": role, "answer": answer, "stance": s.stance, "reason": s.reason}
        for role, answer, s in zip(plan["roles"], answers, stances)
    ]

    # Agreement check in code: the people who know should agree (README), so a
    # unanimous reading is final; a conflict or no usable answer gets reflection.
    votes = {w["stance"] for w in witnesses} - {"no_comment"}
    if votes == {"favourable"}:
        verdict, how = "PASS", "unanimous"
    elif votes == {"unfavourable"}:
        verdict, how = "FAIL", "unanimous"
    elif votes:
        evidence = "\n".join(
            f"- {w['role']}: {w['answer']}\n  first reading: {w['stance']} ({w['reason']})" for w in witnesses
        )
        reflected = await _with_retries(
            lambda: agent["reflect"].ainvoke({**_point_inputs(point), "evidence": evidence})
        )
        verdict, how = reflected.verdict, f"reflection: {reflected.reason}"
    else:
        verdict, how = DEFAULT_VERDICT, "no usable interview answer; default"
    return {"point": point, "verdict": verdict, "how": how, "witnesses": witnesses}


def _fallback_summary(findings: list[dict[str, Any]]) -> str:
    return " ".join(
        f"Point {f['point']} ({POINT_PLANS[f['point']]['name']}): {f['verdict']}, "
        + "; ".join(w["reason"] for w in f["witnesses"] if w["stance"] != "no_comment") + "."
        for f in findings
    )


async def _analyze_company(agent: dict[str, Any], name: str, tools: list[Any]) -> dict[str, Any]:
    interview = next(t for t in tools if t.name == "interview")
    # The three points use different topics, so they run in parallel.
    findings = await asyncio.gather(*(_judge_point(agent, interview, p) for p in CHOSEN_POINTS))
    points = {f["point"]: f["verdict"] for f in findings}

    findings_text = "\n".join(
        f"Point {f['point']} ({POINT_PLANS[f['point']]['name']}): {f['verdict']}\n"
        + "\n".join(f"  - {w['role']}: {w['answer']}" for w in f["witnesses"])
        for f in findings
    )
    try:
        message = await _with_retries(lambda: agent["summary"].ainvoke(
            {"company": name.replace("_", " ").title(), "findings": findings_text}
        ))
        summary = str(message.content).strip() or _fallback_summary(findings)
    except Exception:
        summary = _fallback_summary(findings)
    return {"points": points, "summary": summary}


async def analyze_companies(agent: Any, companies: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """Run your agent and return its verdicts on CHOSEN_POINTS for every company.

    ``companies`` maps each company folder name to its four LangChain tools:

        list_documents()             names and dates of the company's documents
        read_document(name)          the text of one document
        sql_query(query)             one read-only SQLite query on company.db
        interview(role, topic)       one interview answer (INTERVIEW_LIMIT per company)

    Give the tools to a tool-calling agent, e.g.
    ``create_agent(model, tools=companies[name], system_prompt=...)``, or call
    them yourself with ``tool.invoke({...})``. The README explains the
    documents, the database tables and the interview roles and topics.

    Return one dictionary per company folder name, e.g. with
    CHOSEN_POINTS = [1, 7, 15]:

        {
            "hollins_group": {
                "points": {1: "PASS", 7: "FAIL", 15: "PASS"},
                "summary": "Two or three sentences explaining the verdicts.",
            },
            ...
        }

    Each chosen point needs a verdict ("PASS" or "FAIL"; True/False also
    works). Other points are ignored.

    Catch errors per company (e.g. a failed API call) and still return an
    entry for it: an exception here means no results.csv, which scores zero.
    You may analyse companies in parallel (e.g. ``asyncio.gather``).
    """
    ### YOUR CODE HERE
    semaphore = asyncio.Semaphore(MAX_PARALLEL_COMPANIES)

    async def guarded(name: str, tools: list[Any]) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            try:
                return name, await _analyze_company(agent, name, tools)
            except Exception as exc:  # one failed company must not crash the run
                return name, {
                    "points": {p: DEFAULT_VERDICT for p in CHOSEN_POINTS},
                    "summary": f"Analysis failed ({type(exc).__name__}: {exc}); default verdicts given.",
                }

    results = await asyncio.gather(*(guarded(name, tools) for name, tools in companies.items()))
    return dict(results)


# Everything below is provided runner/scoring code. No edits are needed.
# Your code must read company data only through these tools: do not open
# company.db or interviews.dat directly.

INTERVIEWS_USED: dict[str, int] = {}


def company_folders(folder: Path) -> list[Path]:
    """Return one sub-folder per company directly inside *folder*, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def make_tools(company_folder: Path) -> list[Any]:
    """Build the four tools for one company."""
    from langchain_core.tools import tool

    name = company_folder.name
    db_uri = f"{(company_folder / 'company.db').resolve().as_uri()}?mode=ro"
    answers = json.loads(zlib.decompress((company_folder / "interviews.dat").read_bytes()))
    documents = {p.name: p for p in sorted(company_folder.glob("*.md"))}
    lock = threading.Lock()
    INTERVIEWS_USED[name] = 0

    @tool
    def list_documents() -> str:
        """List this company's documents (annual letter, earnings calls, news) with their dates."""
        lines = []
        for doc_name, path in documents.items():
            text = path.read_text(encoding="utf-8").splitlines()
            dated = next((line for line in text if line.startswith("Date:")), "")
            lines.append(f"{doc_name}: {text[0].lstrip('# ')} ({dated or 'undated'})")
        return "\n".join(lines)

    @tool
    def read_document(name: str) -> str:
        """Return the full text of one document named by list_documents, e.g. 'annual_letter_2025.md'."""
        path = documents.get(name.strip())
        if path is None:
            return f"No document named {name!r}. Available: {', '.join(documents)}"
        return path.read_text(encoding="utf-8")

    @tool
    def sql_query(query: str) -> str:
        """Run one read-only SQLite query on this company's database and return at most 50 rows.

        Amounts are in USD thousands. Dates are 'YYYY-MM-DD', months 'YYYY-MM', quarters like '2025Q3'.
        Tables:
          company(name, industry)
          industry_medians(industry, revenue_growth_2y_pct, new_product_share_pct, new_product_sales_per_rd, operating_margin_pct)
          orders(order_id, order_date, product, customer, amount, note)
          products(product, launch_date)
          monthly_costs(month, cost_of_sales, selling_admin, research_development)
          quarterly_cash(quarter, operating_cash_flow, capex, cash_end)
          capex_plan(year, planned_capex)
          returns(month, units_shipped, units_returned)
          insider_trades(trade_date, person, title, side, shares, price)
        Use SUM, GROUP BY and WHERE to compute totals rather than reading raw rows.
        """
        try:
            con = sqlite3.connect(db_uri, uri=True)
            try:
                cursor = con.execute(query)
                columns = [c[0] for c in cursor.description or []]
                rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
            finally:
                con.close()
        except sqlite3.Error as exc:
            return f"SQL error: {exc}"
        if not columns:
            return "Query returned no columns."
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        out = [" | ".join(columns)] + [" | ".join(fmt(v) for v in row) for row in rows[:MAX_SQL_ROWS]]
        if len(rows) > MAX_SQL_ROWS:
            out.append(f"... more rows not shown (limit {MAX_SQL_ROWS}); aggregate or add LIMIT/OFFSET.")
        if not rows:
            out.append("(no rows)")
        return "\n".join(out)

    @tool
    def interview(role: str, topic: str) -> str:
        """Interview one person about the company (at most 10 interviews per company).

        role: customer, current_employee, former_employee, competitor or supplier.
        topic: sales_team, working_conditions, top_management, decision_making or competitive_edge.
        People who would not know about a topic say so; the README lists who knows what.
        """
        role, topic = role.strip().lower(), topic.strip().lower()
        if role not in answers:
            return f"Unknown role {role!r}. Roles: {', '.join(answers)}. (Not counted.)"
        if topic not in answers[role]:
            return f"Unknown topic {topic!r}. Topics: {', '.join(answers[role])}. (Not counted.)"
        with lock:
            if INTERVIEWS_USED[name] >= INTERVIEW_LIMIT:
                return f"Interview limit reached: {INTERVIEW_LIMIT} interviews per company."
            INTERVIEWS_USED[name] += 1
        return answers[role][topic]

    return [list_documents, read_document, sql_query, interview]


def normalize_verdict(value: Any) -> str | None:
    """Accept PASS/FAIL (any case) or a boolean."""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    text = str(getattr(value, "content", value)).strip().upper()
    return text if text in VERDICTS else None


def check_chosen_points(chosen: Any) -> list[int]:
    """CHOSEN_POINTS must name 3 different points between 1 and 15."""
    try:
        points = sorted({int(p) for p in chosen})
    except (TypeError, ValueError):
        points = []
    if len(points) != 3 or len(chosen) != 3 or not all(p in FISHER_POINTS for p in points):
        raise SystemExit("Set CHOSEN_POINTS at the top of hw3.py to 3 different point numbers "
                         f"between 1 and 15, e.g. [1, 7, 15] (now: {chosen!r}).")
    return points


def normalize_points(points: Any, chosen: list[int]) -> dict[int, str | None]:
    """Map each chosen point number to PASS/FAIL, or None when missing/invalid."""
    points = points if isinstance(points, dict) else {}
    by_number: dict[int, Any] = {}
    for key, value in points.items():
        try:
            by_number[int(key)] = value
        except (TypeError, ValueError):
            continue
    return {p: normalize_verdict(by_number.get(p)) for p in chosen}


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read the expected 15 point verdicts per company."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def point_errors(points: dict[int, str | None], expected: dict[str, Any] | None) -> tuple[int, str]:
    """Count correct point verdicts and describe the wrong ones."""
    if expected is None:
        return 0, ""
    wrong = []
    for p in points:
        truth = expected["points"][str(p)]
        if points[p] != truth:
            wrong.append(f"{p}: expected {truth}, predicted {points[p] or 'missing'}")
    if all(v is None for v in points.values()):
        return 0, "all point verdicts missing"
    return len(points) - len(wrong), "; ".join(wrong)


def balanced_point_accuracy(
    analyses: dict[str, Any], truth: dict[str, dict[str, Any]], names: list[str], chosen: list[int]
) -> float:
    """Mean of the accuracy on true-PASS points and on true-FAIL points.

    Most verdicts are PASS, so answering PASS everywhere gets a high plain
    accuracy but only 50% balanced accuracy. Grading uses the balanced number.
    """
    hits = {"PASS": [0, 0], "FAIL": [0, 0]}
    for name in names:
        if name not in truth:
            continue
        entry = analyses.get(name)
        points = normalize_points(entry.get("points") if isinstance(entry, dict) else None, chosen)
        for p in chosen:
            expected = truth[name]["points"][str(p)]
            hits[expected][0] += points[p] == expected
            hits[expected][1] += 1
    rates = [h / n for h, n in hits.values() if n]
    return sum(rates) / len(rates) if rates else 0.0


def write_results(
    names: list[str], analyses: dict[str, Any], truth: dict[str, dict[str, Any]], chosen: list[int]
) -> tuple[Path, int]:
    """Write results.csv and return the number of correct point verdicts."""
    output = Path("results.csv")
    correct_points = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["company", "points", "points_correct", "point_errors", "interviews_used", "summary"])
        for name in names:
            entry = analyses.get(name)
            entry = entry if isinstance(entry, dict) else {}
            points = normalize_points(entry.get("points"), chosen)
            expected = truth.get(name)
            n_points, errors = point_errors(points, expected)
            correct_points += n_points
            writer.writerow([
                name,
                " ".join(f"{p}:{points[p] or 'missing'}" for p in chosen),
                f"{n_points}/{len(chosen)}" if expected else "",
                errors,
                INTERVIEWS_USED.get(name, 0),
                str(entry.get("summary", "")).strip(),
            ])
    return output, correct_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW3 on company data folders")
    parser.add_argument(
        "--data-folder",
        required=True,
        type=Path,
        help="folder containing one sub-folder per company",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    folders = company_folders(folder)
    if not folders:
        raise SystemExit(f"no company folders found in {folder}")

    chosen = check_chosen_points(CHOSEN_POINTS)
    load_env_file()
    companies = {path.name: make_tools(path) for path in folders}
    agent = build_agent()
    analyses = await analyze_companies(agent, companies)
    if not isinstance(analyses, dict):
        raise TypeError("analyze_companies() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct_points = write_results(list(companies), analyses, truth, chosen)
    summary = f" Your points: {', '.join(map(str, chosen))}."
    if truth:
        summary += (
            f" Point accuracy: {correct_points}/{len(companies) * len(chosen)}"
            f" (balanced: {balanced_point_accuracy(analyses, truth, list(companies), chosen):.1%})."
        )
    print(f"Processed {len(companies)} compan{'y' if len(companies) == 1 else 'ies'}. Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.data_folder.is_dir():
        raise SystemExit(f"not a folder: {args.data_folder}")
    return asyncio.run(run(args.data_folder))


if __name__ == "__main__":
    raise SystemExit(main())
