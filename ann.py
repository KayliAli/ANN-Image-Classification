""" problem tanımı: mnist veri seti ile rakam sınıflandırma projesi
"""

#libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms  #görüntü dönüşümleri/vektör
import matplotlib.pyplot as plt #Görselleştirme

# optional : cihazı belirle
device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# veri seti yükleme/data loading
def get_data_loaders(batch_size=64): # Her iteresyonda işlenecek veri miktarı

    transform=transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5,),(0.5,))]) #görüntüyü tensore çevirir
    
    #mnist veri seti indir ve eğtim test kumelerini oluştur
    train_set = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform)
    test_set = torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform)
    
    #pytorch veri yükleyicisi oluştur
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)

    return train_loader,test_loader

train_loader,test_loader=get_data_loaders()

# data visualization
def visualize_samples(loader,n):
    images,labels = next(iter(loader)) # ilk batch den görüntü ve etiketleri alalım
    fig,axes=plt.subplots(1,n,figsize=(10,5))
    for i in range(n):
        axes[i].imshow(images[i].squeeze(),cmap="gray")
        axes[i].set_title(f"Label:{labels[i].item()}")
        axes[i].axis("off")
    plt.show()

visualize_samples(train_loader,4)

# define ANN model
class NoralNetwork(nn.Module): #pythorch'un 'nn.Module' sınıfndan miras alınır

    def __init__(self):
        super(NoralNetwork,self).__init__()

        # Elimizde bulunan görüntüleri vektör haline çevirecegiz
        self.flatten= nn.Flatten()

        self.fc1= nn.Linear(28*28,128)   # ilk tam bağıl katmanı oluştur(Input Layer)
        self.relu=nn.ReLU()# aktivasyon fonksiyonu oluştur
        self.fc2 = nn.Linear(128,64)# ikinci tam baglı katmanı oluştur
        self.fc3 = nn.Linear(64,10)# cıktı katmanını oluştur

    def forward(self,x): #forward propagation ;ileri yayılım x:görüntü(28x28)
        
        x = self.flatten(x)
        x=self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x=self.relu(x)  # Aktivasyon katmanı
        x = self.fc3(x)  #output katmanı

        return x  #modelimizin çıktısı


# create model and complite
model =NoralNetwork().to(device)


#Kayıp fonksiyonu ve optimizasyon fonksiyonları

define_loss_and_optimizer = lambda model:(
    nn.CrossEntropyLoss(), #hata fonksiyonu
    optim.Adam(model.parameters(),lr=0.001) # weights parametresi güncellenir
)
criterion,optimizer = define_loss_and_optimizer(model)

# train

def train_model(model,train_loader,criterion,optimizer,epochs=10):
    
    model.train() # modelimizi eğitim moduna alalım
    train_losses=[] # her bir epoch sonucunda elde edilen loss degerlerini saklamak için bir liste oluştur
    
    for epoch in range(epochs): # belirtilen epoch sayisi kadar eğitim yapalım
        total_loss=0

        for images,labels in train_loader: # tüm eğitim verileri üzerinde iterasyon gerçekleştir
            images,labels=images.to(device),labels.to(device)

            optimizer.zero_grad()# gradyanları sıfırla
    
            pradictions=model(images) # modeli uygula,forward pro.
            loss = criterion(pradictions,labels) # loss hesaplama 
            loss.backward() # geri yayılım yani gradyan hesaplama
            optimizer.step() # weight parametrelirini güncelle

            total_loss +=loss.item()

        awg_loss = total_loss/len(train_loader)  # ortalama kayıp hesaplar
        train_losses.append(awg_loss)
        print(f"Epoch{epoch+1}/{epoch},Loss:{awg_loss:.3f}")

    plt.figure()
    plt.plot(range(1,epochs +1 ),train_losses,marker="o",linestyle="-",label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.legend()
    plt.show()

train_model(model,train_loader,criterion,optimizer,epochs=1)


# test
def test_model(model,test_loader):
    model.eval()
    correct=0  # dogru tahmin sayacı
    total=0  

    with torch.no_grad(): # gradyan hesaplama kapatıldı
        for images,labels in test_loader:
            images,labels = images.to(device),labels.to(device)
            predicition=model(images)
            _, predicition = torch.max(predicition,1)
            total+=labels.size(0)
            correct+=(predicition == labels).sum().item() #dogru ise sayılır
    
    print(f"Test Accuracy :{100*correct/total:.3f}%")

test_model(model,test_loader)

#main

if __name__=="main":
    train_loader,test_loader = get_data_loaders()
    visualize_samples(train_loader,5)
    model =NoralNetwork().to(device)
    criterion,optimizer=define_loss_and_optimizer(model)
    train_model(model,train_loader,criterion,optimizer)
    test_model(model,test_loader)
