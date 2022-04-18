def add_model_args(parser):
    mimounet = parser.add_argument_group("MIMOUnet")
    mimounet.add_argument(
        "--num-res-blocks",
        help="Number of ResNet blocks inside MIMOUnet block",
        default=8,
        type=int,
    )

    return parser
