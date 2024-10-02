from config import get_config
from Learner import face_learner
import argparse

# python train.py -net mobilefacenet -b 200 -w 4
# python train.py -net mobilefacenet -b 200 -w 4 --use-mixup --mixup-alpha 1.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    
    # 训练相关参数
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]", default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]", default='emore', type=str)
    
    # Mixup 相关参数
    parser.add_argument("--use-mixup", action='store_true', help="Use mixup data augmentation")  # 布尔类型参数，是否使用 mixup
    parser.add_argument("--mixup-alpha", type=float, default=1.0, help="Alpha value for mixup")  # mixup 的 alpha 参数

    args = parser.parse_args()

    conf = get_config()

    # 判断使用哪个网络
    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth    

    # 更新配置文件中的其他参数
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode

    # 添加 Mixup 参数到配置中
    conf.use_mixup = args.use_mixup  # 是否使用 mixup
    conf.mixup_alpha = args.mixup_alpha  # mixup 的 alpha 参数

    # 创建 learner 并启动训练
    learner = face_learner(conf)
    learner.train(conf, args.epochs)
