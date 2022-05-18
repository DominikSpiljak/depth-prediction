def add_model_args(parser):
    dpt = parser.add_argument_group("DPT")
    dpt.add_argument(
        "--no-backbone-pretrain",
        action="store_false",
        dest="backbone_pretrained",
    )
    dpt.add_argument(
        "--pretrained-weights",
        help="Path to pretrained weights for DPT hybrid model",
    )
    return parser
