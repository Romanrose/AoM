#!/usr/bin/env python
"""
AoM (Aspect-oriented Information for Multimodal Aspect-Based Sentiment Analysis) 主入口脚本
基于VLP-MABSA框架改进，支持多模态方面情感分析
"""

import argparse
import sys
import os
from pathlib import Path

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from global_var import (
    global_update,
    twitter15_info_path,
    twitter17_info_path,
    trc_info_path,
    bart_model_dir,
    train15_ckpt_dir,
    train17_ckpt_dir,
    train_trc_ckpt_dir,
    twitter15_log_dir,
    twitter17_log_dir,
)


def main():
    parser = argparse.ArgumentParser(
        description='AoM 多模态方面情感分析框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的的任务:
  twitter15      - Twitter15 数据集训练 (AESC联合任务)
  twitter17      - Twitter17 数据集训练 (AESC联合任务)
  pretrain_trc   - TRC预训练任务
  test           - 测试模式

示例:
 c
  python run_aom.py --task twitter17 --lr 7.5e-5 --batch_size 16
  python run_aom.py --task pretrain_trc --dataset TRC
  python run_aom.py --task twitter15 --do_test --model_path checkpoints/AoM-ckpt/Twitter2015/AoM.pt
        """
    )

    # 任务选择
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['twitter15', 'twitter17', 'pretrain_trc', 'test'],
        help='选择要运行的任务'
    )

    # 添加通用参数 (这些会传递给具体的训练脚本)
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='数据集名称 (twitter15, twitter17, TRC)'
    )

    parser.add_argument('--lr', type=float, default=7e-5, help='学习率')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=35, help='训练轮数')
    parser.add_argument('--gpu_num', type=int, default=1, help='GPU数量')
    parser.add_argument('--rank', type=int, default=0, help='GPU排名 (0-7)')
    parser.add_argument('--no_train', action='store_true', help='只测试，不训练')
    parser.add_argument('--do_test', action='store_true', help='测试模式')

    # 添加其他常用参数
    parser.add_argument('--warmup', type=float, default=0.1, help='预热步数比例')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='梯度裁剪')
    parser.add_argument('--seed', type=int, default=66, help='随机种子')
    parser.add_argument('--model_config', type=str, default='configs/pretrain_base.json', help='模型配置')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')

    # 预训练相关
    parser.add_argument('--trc_pretrain_file', type=str,  default='checkpoints/pytorch_model.bin', help='TRC预训练模型路径')

    # 测试相关
    parser.add_argument('--model_path', type=str, help='测试用的模型路径')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点目录')

    # 解析参数
    args = parser.parse_args()

    # 更新全局变量
    args = global_update(args)

    # 构建MAESC_training.py的参数列表
    cmd_args = ['python', str(Path(__file__).parent / 'MAESC_training.py')]

    # 根据任务设置dataset和路径（符合原始AoM设计）
    if args.task == 'twitter15':
        if not args.dataset:
            args.dataset = 'twitter15'
        cmd_args.extend(['--dataset', 'twitter15', twitter15_info_path])
        cmd_args.extend(['--checkpoint_dir', train15_ckpt_dir])
        # 统一日志目录到 logs/ 下
        if args.log_dir == 'logs':  # 只有在未自定义时才使用默认值
            args.log_dir = twitter15_log_dir
    elif args.task == 'twitter17':
        if not args.dataset:
            args.dataset = 'twitter17'
        cmd_args.extend(['--dataset', 'twitter17', twitter17_info_path])
        cmd_args.extend(['--checkpoint_dir', train17_ckpt_dir])
        # 统一日志目录到 logs/ 下
        if args.log_dir == 'logs':  # 只有在未自定义时才使用默认值
            args.log_dir = twitter17_log_dir
    elif args.task == 'pretrain_trc':
        if not args.dataset:
            args.dataset = 'TRC'
        cmd_args.extend(['--dataset', 'TRC', trc_info_path])
        cmd_args.extend(['--checkpoint_dir', train_trc_ckpt_dir])
        # TRC预训练使用默认log_dir

    # 添加其他参数
    cmd_args.extend([
        '--model_config', args.model_config,
        '--log_dir', args.log_dir,
        '--lr', str(args.lr),
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--warmup', str(args.warmup),
        '--grad_clip', str(args.grad_clip),
        '--seed', str(args.seed),
        '--gpu_num', str(args.gpu_num),
        '--rank', str(args.rank),
        '--trc_pretrain_file', args.trc_pretrain_file,
        '--bart_model', bart_model_dir,
        '--nn_attention_on',
        '--nn_attention_mode', '0',
        '--trc_on',
        '--gcn_on',
        '--dep_mode', '2',
        '--sentinet'
    ])

    # 测试模式
    if args.no_train:
        cmd_args.append('--no_train')

    if args.do_test and args.model_path:
        cmd_args.extend(['--do_test', '--model_path', args.model_path])

    # === GPU设备检查 ===
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\n{'='*60}")
        print(f"GPU环境检查:")
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  可用GPU数量: {gpu_count}")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"{'='*60}\n")

        # 检查rank参数
        if args.rank >= gpu_count:
            print(f"⚠️  警告: rank={args.rank} 超出GPU数量({gpu_count})，自动调整为0")
            args.rank = 0
            cmd_args[cmd_args.index('--rank') + 1] = '0'  # 更新命令中的rank值
            print(f"✅ 已将rank调整为: {args.rank}\n")
        else:
            print(f"✅ 使用GPU {args.rank}: {torch.cuda.get_device_name(args.rank)}\n")
    else:
        print("⚠️  GPU不可用，使用CPU训练\n")

    # 打印命令
    print("=" * 80)
    print("🚀 Running AoM Training:")
    print("=" * 80)
    print("Task:", args.task)
    print("Dataset:", args.dataset)
    print("Command:", ' '.join(cmd_args))
    print("=" * 80)

    # 执行命令
    os.system(' '.join(cmd_args))


if __name__ == '__main__':
    main()
