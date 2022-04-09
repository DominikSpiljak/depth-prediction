def add_model_args(parser):
    laddernet = parser.add_argument_group("LadderNet")
    print(type(laddernet), laddernet.add_argument)
    laddernet.add_argument(
        "--dummy-laddernet",
        help="Dummy argument for testing",
        default="laddernet_placeholder",
    )
    return parser
