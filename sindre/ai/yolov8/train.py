import  torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sindre.ai.utils import set_global_seeds,load_checkpoint,save_checkpoint
import os
import argparse
from components.datasets import IMGDataset
from models import  YOLOV8
from tqdm import tqdm
from components.losses import v8DetectionLoss

def train():
    """Main execution pipeline. """
    os.environ["DISPLAY"]=":0"
    os.chdir(os.path.dirname(__file__))
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_path", default="origin_1067_with_faces.db")
    parser.add_argument("--num_iters", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_name",  default=r"best.pt")
    args = parser.parse_args()
    # 设置随机数
    set_global_seeds(1024)
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")


    # Initialize components
    datasets=IMGDataset(data_dir=args.datasets_path,batch_size=args.batch_size,num_workers=args.num_workers)
    datasets_val=IMGDataset(path=args.datasets_path,batch_size=args.batch_size,num_workers=args.num_workers,val=True)
    print("datasets:",len(datasets))
    print("datasets_val:",len(datasets_val))
    train_dataloader=datasets.train_dataloader()
    val_dataloader=datasets_val.val_dataloader()



    # Initialize optimization
    model = YOLOV8(input_channels=3,num_classes=20,model_name="n",task="Detect").to(device)
    my_loss = v8DetectionLoss(model)
    #optimizer =torch.optim.AdamW(model.parameters(), lr=args.learning_rate,weight_decay=0.005)
    bn_list, weight_list, bias_list = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):
            bias_list.append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d) or "bn" in k:
            bn_list.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):
            weight_list.append(v.weight)
    optimizer= torch.optim.SGD(bn_list, 1e-4, momentum =  0.937, nesterov=True)
    optimizer.add_param_group({"params": weight_list, "weight_decay": 5e-4})
    optimizer.add_param_group({"params": bias_list})
    curr_epoch, best_loss, extra_info = load_checkpoint(args.model_name, model,optimizer)


    # 如果有预训练模型，那么就冻结主干，训练其他，然后在整体训练
    # for param in model.backbone.parameters():
    #     param.requires_grad = False


    # Training loop
    for epoch in range(args.num_iters)[curr_epoch:]:
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, data in enumerate(progress_bar):
            images,labels = data
            optimizer.zero_grad()
            outputs = model(images)
            loss = my_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            train_loss += loss.item()




        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for  data in val_dataloader:
                images,labels = data
                optimizer.zero_grad()
                outputs = model(images)
                loss_val = my_loss(outputs, labels)
                val_loss += loss_val.item()

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)


        # 打印训练信息
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}| LR: {optimizer.param_groups[0]['lr']}")
        save_checkpoint(args.model_name, model,avg_val_loss, optimizer,epoch)


