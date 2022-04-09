def add_model_args(parser):
    dpt = parser.add_argument_group("DPT")
    dpt.add_argument(
        "--dummy-DPT",
        help="Dummy argument for testing",
        default="DPT_placeholder",
    )
    return parser
