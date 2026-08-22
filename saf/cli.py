"""SAF v4 CLI — foundation + backtest + rubric caching."""
import argparse, json, sys, time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from . import config, store, data

console = Console()

def cmd_init(_): store.init(); console.print("[green]✅[/green] saf.db initialized")

def cmd_validate(_):
    cfg = config.load(); th = cfg["settings"]["score_thresholds"]
    console.print(f"[green]✅[/green] Config valid — {len(cfg['baskets'])} baskets, "
                  f"{len(config.all_tickers(cfg))} unique tickers")
    console.print(f"   thresholds: candidate≥{th['candidate']}, watch≥{th['watch']}")

def cmd_fetch(args):
    store.init()
    if args.ticker:
        ok = data.refresh_ticker(args.ticker.upper())
        _print_quality([data.quality_report(args.ticker.upper())])
        sys.exit(0 if ok else 1)
    summary = data.refresh_universe(with_fundamentals=args.fundamentals)
    console.print(f"[green]✅ {summary['ok']}/{summary['tickers']} refreshed, "
                  f"{summary['fail']} failed[/green]")
    if summary["failures"]:
        console.print(f"[red]Failed: {', '.join(summary['failures'])}[/red]")

def cmd_quality(_):
    store.init()
    reports = [data.quality_report(t) for t in config.all_tickers()]
    _print_quality(reports)
    bad = [r for r in reports if not r["usable"]]
    if bad:
        console.print(f"[yellow]⚠️ {len(bad)} tickers excluded by quality gate[/yellow]")

def _parse_flags(raw):
    if isinstance(raw, list): return raw
    try:
        parsed = json.loads(raw or "[]")
        return parsed if isinstance(parsed, list) else []
    except Exception: return []

def _print_quality(reports):
    tbl = Table(title="Data Quality Gate")
    tbl.add_column("Ticker", style="bold cyan")
    tbl.add_column("Usable", justify="center")
    tbl.add_column("Bars", justify="right")
    tbl.add_column("Last", justify="right")
    tbl.add_column("Stale d", justify="right")
    tbl.add_column("0-vol d", justify="right")
    tbl.add_column("Flags", overflow="fold")
    for r in sorted(reports, key=lambda x: x["ticker"]):
        usable = "[green]✅[/green]" if r["usable"] else "[red]❌[/red]"
        flags = _parse_flags(r["flags"])
        flag_txt = ", ".join(flags) if flags else "[dim]—[/dim]"
        tbl.add_row(r["ticker"], usable, str(r["bars"]), str(r["last_date"] or "—"),
                    str(r["stale_days"] if r["stale_days"] is not None else "—"),
                    str(r["zero_vol_days"] if r["zero_vol_days"] is not None else "—"),
                    flag_txt)
    console.print(tbl)

def cmd_prices(args):
    df = store.load_prices(args.ticker.upper())
    if df.empty:
        console.print(f"[red]No stored prices for {args.ticker}. "
                      f"Run: fetch --ticker {args.ticker}[/red]")
        sys.exit(1)
    console.print(f"[bold]{args.ticker.upper()}[/bold] — {len(df)} bars stored")
    console.print(df.tail(args.tail)[["px", "volume"]].to_string())

def cmd_verify_audit(_):
    ok = store.verify_audit_chain()
    console.print("[green]✅ Audit chain intact[/green]" if ok
                  else "[red]❌ AUDIT CHAIN BROKEN[/red]")
    sys.exit(0 if ok else 1)

def cmd_backtest(args):
    store.init()
    from .quant import backtest
    console.print(f"[dim]Walk-forward validation — Score v2, "
                  f"{args.eval_horizon}d horizon, net of costs...[/dim]")
    try:
        out = backtest.run_backtest(step_days=args.step, cost_bps=args.cost_bps,
                                    eval_horizon=args.eval_horizon)
    except RuntimeError as e:
        console.print(f"[red]❌ {e}[/red]")
        sys.exit(1)
    _render_backtest(out["results"])

def _render_backtest(res):
    q = res.get("quintiles_annual_pct")
    if q is None:
        console.print(f"[yellow]{res['verdict_text']}[/yellow]")
        return
    tbl = Table(title=f"Score v2 — annualized fwd return by quintile "
                      f"({res.get('eval_horizon')}d horizon, net of costs)")
    tbl.add_column("Quintile")
    tbl.add_column("Ann. return %", justify="right")
    for i, v in q.items():
        tag = " LOW" if i == 0 else (" HIGH" if i == len(q) - 1 else "")
        tbl.add_row(f"Q{i+1}{tag}", f"{v:+.2f}")
    console.print(tbl)
    c = "green" if res["q5_minus_q1_annual"] >= 3 else "red"
    console.print(f"  Q5−Q1 annualized: [{c}]{res['q5_minus_q1_annual']:+.2f}%[/{c}] (bar: ≥3.0)")
    console.print(f"  IC mean: {res['ic_mean']} | IC IR: {res['ic_ir']} (bar: ≥0.30) | "
                  f"monotonic: {'✅' if res['monotonic'] else '❌'} | turnover: {res['turnover_est']}")
    console.print(f"  sample: {res['n_dates']} rebalances, {res['n_obs']} ticker-dates")
    d = res.get("decay", {})
    console.print("  decay curve (Q5−Q1 %): " + "  ".join(
        f"{h}d: {v:+.2f}%" if v is not None else f"{h}d: n/a" for h, v in d.items()))
    for reg, r in res.get("by_regime", {}).items():
        if r.get("q5_minus_q1_annual") is not None:
            cc = "green" if r["q5_minus_q1_annual"] >= 3 else "red"
            console.print(f"  regime {reg}: spread [{cc}]{r['q5_minus_q1_annual']:+.2f}%[/{cc}], "
                          f"IC IR {r['ic_ir']}")
    ok = res["status"] == "VALIDATED"
    style = "bold green" if ok else "bold red"
    console.print(Panel(
        f"[{style}]{'✅ VALIDATED' if ok else '❌ ' + res['status']}[/{style}]\n"
        + res["verdict_text"],
        title="Scorecard verdict", box=box.DOUBLE,
        border_style="green" if ok else "red"))

def cmd_cache_rubric(args):
    """Pre-warm the SQLite rubric cache. Full diagnostics on failure —
    shows which model was used and the real Groq error."""
    store.init()
    from .ai import evidence, rubric

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        cfg = config.load()
        tickers = config.all_tickers(cfg)
        if args.top:
            tickers = tickers[:args.top]

    console.print(f"[dim]Warming rubric cache for {len(tickers)} tickers "
                  f"(delay {args.delay}s between calls)...[/dim]")
    saved = failed = skipped = 0

    for i, t in enumerate(tickers, 1):
        if store.get_cached_rubric(t) and not args.force:
            console.print(f"  [{i}/{len(tickers)}] {t}: [green]cached[/green]")
            skipped += 1
            continue

        console.print(f"  [{i}/{len(tickers)}] {t}: scoring...", end=" ")
        try:
            pack = evidence.build_evidence_pack(t)
        except Exception as e:
            console.print(f"[red]evidence failed: {str(e)[:80]}[/red]")
            failed += 1
            continue

        if not pack.get("business_desc"):
            console.print("[yellow]no evidence (skipped)[/yellow]")
            failed += 1
            continue

        res = rubric.score_bottleneck(t, pack)
        if "error" not in res:
            store.save_rubric(t, res["total"], res)
            meta = res.get("llm_meta", {})
            console.print(f"[green]saved ({res['total']}/30)[/green] "
                          f"[dim]via {meta.get('model', '?')}[/dim]")
            saved += 1
        else:
            # ── FULL error display: model + real error + raw output ──
            dbg = res.get("debug", {})
            console.print("[red]FAILED[/red]")
            console.print(f"      model: {dbg.get('model', '?')}")
            console.print(f"      error: {dbg.get('error', res.get('error', 'unknown'))}")
            if dbg.get("raw"):
                console.print(f"      raw:   {dbg['raw'][:200]}")
            failed += 1
        time.sleep(args.delay)

    console.print(f"\n[bold]Summary:[/bold] [green]{saved} saved[/green], "
                  f"[dim]{skipped} already cached[/dim], [red]{failed} failed[/red]")
    if saved:
        console.print("[dim]These scores now feed the B-component in score_v2 and "
                      "the backtest automatically.[/dim]")

def main():
    p = argparse.ArgumentParser(prog="saf", description="Skia Alpha Fund v4")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create database schema").set_defaults(fn=cmd_init)
    sub.add_parser("validate", help="validate universe.yaml").set_defaults(fn=cmd_validate)

    f = sub.add_parser("fetch", help="incremental data refresh")
    f.add_argument("--ticker", help="single ticker")
    f.add_argument("--fundamentals", action="store_true",
                   help="also refresh fundamentals for basket holdings")
    f.set_defaults(fn=cmd_fetch)

    sub.add_parser("quality", help="data quality gate report").set_defaults(fn=cmd_quality)

    pr = sub.add_parser("prices", help="show stored prices")
    pr.add_argument("ticker")
    pr.add_argument("--tail", type=int, default=10)
    pr.set_defaults(fn=cmd_prices)

    sub.add_parser("verify-audit", help="verify hash-chained audit log").set_defaults(fn=cmd_verify_audit)

    bt = sub.add_parser("backtest", help="walk-forward validation of Score v2")
    bt.add_argument("--step", type=int, default=21, help="rebalance spacing (trading days)")
    bt.add_argument("--cost-bps", type=float, default=10.0, help="one-side cost in bps")
    bt.add_argument("--eval-horizon", type=int, default=63, choices=[21, 63, 126],
                    help="forward-return horizon to grade (days)")
    bt.set_defaults(fn=cmd_backtest)

    cr = sub.add_parser("cache-rubric", help="pre-warm AI rubric cache")
    cr.add_argument("--tickers", help="comma-separated list, e.g. APD,LIN,MKSI")
    cr.add_argument("--top", type=int, help="score the first N universe tickers alphabetically")
    cr.add_argument("--force", action="store_true", help="re-score even if cached")
    cr.add_argument("--delay", type=float, default=3.0, help="seconds between LLM calls")
    cr.set_defaults(fn=cmd_cache_rubric)

    args = p.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()