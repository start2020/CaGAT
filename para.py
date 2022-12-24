import argparse

def parameter():
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="tfinance",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--model_name", type=str, default="BWGNN") # BWGNN
    parser.add_argument("--train_ratio", type=float, default=0.01, help="Training ratio") # amazon-0.4, yelp-0.01, tfinance-0.01
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs") # 100
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--GCN_train", type=bool, default=False)
    #parser.add_argument("--LOSS", type=str, default="CrossEntropy") # CrossEntropy, CaGCNLoss, NewLoss,Loss_Ano,Loss_Lab,Loss_Dis
    parser.add_argument("--cal_models_list", type=str, default="1,2") # [0,1,2,3,4,5,6]["BWGNN","CaGCN","CaGAT","His","Iso","BBQ","TS"]
    parser.add_argument("--bins_list", type=str, default="15") #5,10,15

    parser.add_argument("--GCN_hidden_dim", type=int, default=16)
    parser.add_argument("--GCN_lr", type=float, default=0.01  )  # 0.01
    parser.add_argument("--GCN_weight_decay", type=float, default=5e-4)
    parser.add_argument("--GCN_epochs", type=int, default=200)
    args = parser.parse_args()
    return args

