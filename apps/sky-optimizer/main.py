"""CPU spectral fitting against independently generated reference datasets."""
import sys
import runpy

COMMANDS = {"queries": "make_wavelength_search_queries", "search": "search_wavelengths_cpu",
            "counts": "compare_wavelength_counts_cpu", "preview": "preview_wavelengths_cpu"}
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python apps/sky-optimizer/main.py {queries|search|counts|preview} [options]")
        raise SystemExit(0 if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help") else 2)
    command = sys.argv.pop(1)
    runpy.run_module(COMMANDS[command], run_name="__main__")
