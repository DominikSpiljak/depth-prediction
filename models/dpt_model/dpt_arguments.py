def add_model_args(parser):
    dpt = parser.add_argument_group("DPT")
    dpt.add_argument(
        "--no-pretrain",
        action="store_false",
        dest="pretrained",
    )
    return parser
