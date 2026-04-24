try:
    from .draw import main
except ImportError:  # pragma: no cover
    from draw import main


if __name__ == "__main__":
    main()
