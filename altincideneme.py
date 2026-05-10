from torch.amp import GradScaler, autocast
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import StepLR 


class FractureDataset(Dataset):
    
    def __init__(self,img_dir,label_dir,transform=None):
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.transform=transform
        self.image_filenames=sorted(os.listdir(img_dir))
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self,idx):
        
        img_name=self.image_filenames[idx]
        img_path=os.path.join(self.img_dir,img_name)
        image=Image.open(img_path)
        
        if self.transform:
            image=self.transform(image)
            
        label_name=os.path.splitext(img_name)[0]+'.txt'
        label_path=os.path.join(self.label_dir,label_name)
        
        final_label=0
        
        if os.path.exists(label_path):
            with open(label_path,'r') as f:
                for line in f:
                    class_id=int(line.strip().split()[0])
                    
                    if(class_id==3):
                        final_label=1
                        break
        
        return image,torch.tensor(final_label,dtype=torch.long)
    

def get_dataLoaders(batch_size):
    
    transform_train=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(416,padding=4),
        transforms.RandomRotation(10), #görüntüleri rastgele +10 ile -10 derece döndürüyoruz
        transforms.RandomPerspective(distortion_scale=0.05,p=0.5), #Hafif perspektif bozulması
        transforms.ColorJitter(brightness=0.1,contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    
    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])  

    train_img_dir = r"C:\Users\alpcu\Desktop\verionisleme\GRAZPEDWRI-DX_dataset\GRAZPEDWRI-DX_dataset\images\train_aug_processed"
    train_label_dir = r"C:\Users\alpcu\Desktop\verionisleme\GRAZPEDWRI-DX_dataset\GRAZPEDWRI-DX_dataset\labels\train_aug_processed"
    
    test_img_dir = r"C:\Users\alpcu\Desktop\verionisleme\GRAZPEDWRI-DX_dataset\GRAZPEDWRI-DX_dataset\images\test_processed"
    test_label_dir = r"C:\Users\alpcu\Desktop\verionisleme\GRAZPEDWRI-DX_dataset\GRAZPEDWRI-DX_dataset\labels\test_processed"
    
    train_dataset=FractureDataset(img_dir=train_img_dir,label_dir=train_label_dir,transform=transform_train)
    test_dataset=FractureDataset(img_dir=test_img_dir,label_dir=test_label_dir,transform=transform_test)
    
    train_loader=DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)
    test_loader=DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)
    
    siniflar=["YOK","VAR"]
    
    return train_loader,test_loader,siniflar


def visualization(n,images,labels,class_names):
    
    if images.is_cuda:
        images = images.cpu()
        labels = labels.cpu()
        
    images=images*0.5+0.5
    np_img=images.numpy()
    
    fig,ax=plt.subplots(1,n,figsize=(15,2))
    fig.suptitle(f"İlk {n} Görüntü", fontsize=16)
    
    is_single_image=n==1
    
    for i in range(n):
        img=np.transpose(np_img[i],(1,2,0))
        
        if is_single_image:
            current_ax=ax
        else:
            current_ax=ax[i]
        
        if img.shape[2]==1:
            img=img.squeeze(axis=2)
        
        current_ax.imshow(img,cmap='gray')
        label_index=int(labels[i].item()) 
        
        current_ax.set_title(class_names[label_index], color="red" 
                              if label_index==1 else "green")
        current_ax.axis('off')
    
    plt.tight_layout()
    plt.show()

class KirikVarmiYokmu(nn.Module):
    
    def __init__(self):
        super(KirikVarmiYokmu,self).__init__()
        
        #Evrisim Katmanlari (özellik)
        #... (Blok 1-4 tanımlamaları)
        
        #1. blok
        self.conv1=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2) 
        
        #2. blok
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=1)
        self.bn2=nn.BatchNorm2d(128)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        
        #3. blok
        self.conv3=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=1)
        self.bn3=nn.BatchNorm2d(256)
        self.relu3=nn.ReLU()
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2)
        
        #4. blok
        self.conv4=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=1)
        self.bn4=nn.BatchNorm2d(512)
        self.relu4=nn.ReLU()
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv5=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1)
        self.bn5=nn.BatchNorm2d(512)
        self.relu5=nn.ReLU()
        self.pool5=nn.MaxPool2d(kernel_size=2,stride=2)
    
        #OPTİMİZASYON: GAP ekleniyor
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.flatten=nn.Flatten()
        
        #TAM BAĞLANTILI KATMANLAR (karar vericiler)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.5)

        # (Çünkü 256*26*26 yerine Global Average Pooling kullanılıyor)
        self.fc1=nn.Linear(in_features=512,out_features=1024)
        self.bn6=nn.BatchNorm1d(1024)

        self.fc2=nn.Linear(in_features=1024,out_features=512)
        self.bn7=nn.BatchNorm1d(512)

        self.fc3=nn.Linear(in_features=512,out_features=2)

    
    def forward(self,x):
        
        #1. blok
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        
        #2. blok
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.pool2(x)

        #3. blok
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu3(x)
        x=self.pool3(x)

        #4. blok
        x=self.conv4(x)
        x=self.bn4(x)
        x=self.relu4(x)
        x=self.pool4(x)
        
        x=self.conv5(x)
        x=self.bn5(x)
        x=self.relu5(x)
        x=self.pool5(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1) # x.view(Batch_size, 256) boyutuna gelir
        
        #karar vericiler
        x=self.fc1(x)
        x=self.bn6(x)
        x=self.relu(x)
        x=self.dropout(x)
        
        x=self.fc2(x)
        x=self.bn7(x)
        x=self.relu(x)
        
        #FC3 (son katman)
        x=self.fc3(x)

        return x

def train_model(model,train_loader,criterion,optimizer,num_epochs,device,scheduler=None):
    
    print("egitim başladi\n")
    
    model.train()
    
    train_losses=[]
    train_accuracies=[]
    
    scaler = GradScaler(enabled=True) 
    
    for epoch in range(num_epochs):
        
        total_loss=0
        correct=0
        total_samples=0
        
        for images,labels in train_loader:
            images=images.to(device)
            labels=labels.to(device).long()
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                predictions=model(images)
                loss=criterion(predictions,labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss+=loss.item()*images.size(0)
            _, predicted=torch.max(predictions.data,1) 
            correct+=(predicted==labels).sum().item()
            total_samples+=labels.size(0)
        
        avg_loss=total_loss/total_samples
        accuracy=correct/total_samples
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy*100)
        
        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Kayip: {avg_loss:.4f}, Dogruluk: {accuracy*100:.2f}% ,LR: {optimizer.param_groups[0]['lr']:.6f}")
        
    return train_losses,train_accuracies


def kayipvedogrulukgrafigi(train_losses,train_accuracies):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses,label='Training Loss')
    plt.title('Eğitim Loss Değişimi')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Eğitim Accuracy Değişimi')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
def test_model(model, test_loader, device, dataset_type, class_names=['Kirik Yok', 'Kirik Var']):
    
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    test_accuracy = 100 * test_correct / test_total
    print(f"\n {dataset_type} Doğruluğu = {test_accuracy:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names, zero_division=0))
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Konfüzyon Matrisi ({dataset_type})')
    plt.ylabel('Gerçek Sınıf (True Label)')
    plt.xlabel('Tahmin Edilen Sınıf (Predicted Label)')
    plt.show()
    
    return test_accuracy

if __name__=='__main__':
    
    torch.backends.cudnn.benchmark = True
    
    device=torch.device("cuda")
    print("Kullanilan cihaz= ",device)
    
    batch_size=16
    train_loader,test_loader,siniflar=get_dataLoaders(batch_size)
    
    dataiter=iter(train_loader)
    images,labels=next(dataiter)
    
    
    print("veri setinin boyutu= ",images.shape)
    print("egitim veri seti boyutu(batch boyutunda)= ",len(train_loader))
    print("test veri seti boyutu(batch boyutunda)= ",len(test_loader))
    print("siniflar= ",siniflar)

    visualization(10,images,labels,siniflar)

    model=KirikVarmiYokmu().to(device)
    print("model olusuturuldu\n")
    
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    
    scheduler=StepLR(optimizer,step_size=10,gamma=0.5) #optimizer= hangi alogritmaya göre işleyeceği
    #step_size = kaç adımda bir değişeceği
    #gamma= öğrenme oranını azaltmak için kullanılır

    train_losses,train_accuracies=train_model(model,
                                              train_loader,
                                              criterion,
                                              optimizer,
                                              40,
                                              device,
                                              scheduler=scheduler)
    
    final_train_accuracy = train_accuracies[-1] 
    print(f"\nEgitim Tamamlandi. Final Egitim Dogrulugu: {final_train_accuracy:.2f}%")
    
    kayipvedogrulukgrafigi(train_losses,train_accuracies)

    
    test_accuracy=test_model(model,test_loader,device,dataset_type="test")
    train_accuracy=test_model(model,train_loader,device,dataset_type="train")
    
    #model çıktısı kaydetme
    
    model_kayit_yolu="altinci_deneme_kirik_tespit.pth"
    
    torch.save(model.state_dict(),model_kayit_yolu)
    
    print(f"model başarıyla kaydedildi= {model_kayit_yolu}")