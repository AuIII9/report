"""Default: graphical interface. Use --cli for the command-line interface."""
import sys


if __name__ == "__main__":
    if "--cli" in sys.argv:
        from main_cli import main
        main([arg for arg in sys.argv[1:] if arg != "--cli"])
    else:
        try:
            from gui import main
            main()
        except ImportError as exc:
            print(f"图形界面不可用：{exc}\n可运行 python main.py --cli，或安装带 Tk 的 Python。")
            sys.exit(1)
