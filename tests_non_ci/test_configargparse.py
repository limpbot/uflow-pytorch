from usflow import options


def main():
    parser = options.setup_comon_options()
    args = parser.parse_args()

    a = 0

    vars(args)


if __name__ == "__main__":
    main()
