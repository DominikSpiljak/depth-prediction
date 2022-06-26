def add_model_args(parser):
    laddernet = parser.add_argument_group("LadderNet")
    laddernet.add_argument(
        "--densenet",
        help="Which densenet to use",
        default="densenet121",
        choices=[
            "densenet121",
            "densenet161",
            "densenet169",
            "densenet201",
        ],
    )
    laddernet.add_argument(
        "--no-pretrain",
        action="store_false",
        dest="pretrained",
    )
    laddernet.add_argument(
        "--use-checkpointing",
        action="store_true",
    )
    return parser
