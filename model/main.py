import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def create_model(data):
    NUM_CLASSES = 1
    NUM_FEATURES = 9
    batch_size=32
    scaler=StandardScaler()
    X=data.drop(["Survived"],axis=1)
    y=data["Survived"]
    X = scaler.fit_transform(X)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y.values).float().unsqueeze(1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    def accuracy_fn(y_true, y_pred):
        y_true=y_true.squeeze().long()
        y_pred=y_pred.squeeze().long()
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc
    
    class CancerModel(nn.Module):
        def __init__(self, input_features, output_features, hidden_units):
            super().__init__()
            self.linear_layer_stack = nn.Sequential(
                nn.Linear(in_features=input_features, out_features=hidden_units),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=hidden_units, out_features=hidden_units),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=hidden_units, out_features=hidden_units),
                nn.ReLU(),
                nn.Linear(in_features=hidden_units, out_features=output_features),
            )
        def forward(self, x):
            return self.linear_layer_stack(x)

    model_1 = CancerModel(input_features=NUM_FEATURES,output_features=NUM_CLASSES,hidden_units=32)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model_1.parameters(), lr=0.01)
    torch.manual_seed(42)
    epochs = 60

    # Training loop with batch-wise processing
    for epoch in range(epochs):
        model_1.train()
        train_loss = 0
        train_acc = 0
        for batch_X, batch_y in train_loader:
            y_logits = model_1(batch_X).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))
            loss = loss_fn(y_logits, batch_y.squeeze())
            acc = accuracy_fn(y_true=batch_y, y_pred=y_pred)
            train_loss += loss.item()
            train_acc += acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        model_1.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                test_logits = model_1(batch_X).squeeze()
                test_pred = torch.round(torch.sigmoid(test_logits))
                loss = loss_fn(test_logits, batch_y.squeeze())
                acc = accuracy_fn(y_true=batch_y, y_pred=test_pred)
                test_loss += loss.item()
                test_acc += acc
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        # if epoch % 10 == 0:
        #     print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

    return model_1,scaler

def get_clean_data(path):
    data=pd.read_csv(path)
    data["Sex"]=data['Sex'].map({'male':1,'female':0})
    data["Embarked"]=data['Embarked'].map({'S':0,'C':1, 'Q':2})
    data['Cabin'] = data['Cabin'].isnull().astype(float)
    data=data.drop(['PassengerId','Name','Ticket'],axis=1)
    data['Age']=data['Age'].fillna(data['Age'].median())
    data['Embarked']=data['Embarked'].fillna(data['Embarked'].median()) 
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data = data.astype(float)
    return data

def test_preprocess(path,scaler):
    data=pd.read_csv(path)
    data["Sex"] = data['Sex'].map({'male': 1, 'female': 0})
    data["Embarked"] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    data['Cabin'] = data['Cabin'].isnull().astype(float)
    data = data.drop(['Name', 'Ticket'], axis=1)
    data['Age'] = data.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.mean()))
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].median())
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    X = data.drop(['PassengerId'], axis=1)
    X = scaler.transform(X)
    X = torch.from_numpy(np.array(X)).float()
    return X, data['PassengerId']

def main():
    data_train=get_clean_data("data/train.csv")
    model,scaler=create_model(data_train)
    X_test,pass_id=test_preprocess("data/test.csv",scaler)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits)).int()
    submission = pd.DataFrame({
        'PassengerId': pass_id,
        'Survived': test_pred.numpy()
    })
    submission.to_csv("data/submission.csv", index=False)
    print("Submission file saved as submission.csv")

if __name__=='__main__':
    main()