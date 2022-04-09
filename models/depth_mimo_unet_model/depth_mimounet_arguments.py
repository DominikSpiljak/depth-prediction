def add_model_args(parser):
    mimounet = parser.add_argument_group("MIMOUnet")
    print(type(mimounet), mimounet.add_argument)
    mimounet.add_argument(
        "--min-depth",
        help="Minimum that a depth can be in ground truth maps",
        default=0,
        type=float,
    )

    mimounet.add_argument(
        "--max-depth",
        help="Maximum that a depth can be in ground truth maps",
        default=10,
        type=float,
    )

    mimounet.add_argument(
        "--num-res-blocks",
        help="Number of ResNet blocks inside MIMOUnet block",
        default=8,
        type=int,
    )

    return parser
