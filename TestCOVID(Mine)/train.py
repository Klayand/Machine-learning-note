from model import *

train_path = './data/covid.train.csv'
test_path = './data/covid.test.csv'


def train(train_loader, valid_loader, model,
          lr=1e-4, weight_decay=1e-4, total_epcohs=150):
    '''Train your data while validate your data '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_loss = 99999
    valid_loss_record = []
    train_loss_record = []

    if os.path.exists('model.ckpt'):
        model.load_state_dict(torch.load('model.ckpt'))

    from tqdm import tqdm
    for epoch in tqdm(range(1, total_epcohs + 1),colour='blue'):
        # train part
        model.train()  # Set your model into training
        total_loss = 0
        for x, y in train_loader:
            x = model(x)   # x is a tensor, whose size is [2,1]  (2 is row, 1 is column)
            x = x.squeeze()  # using squeeze, x becomes [2], which only has one row
            loss = criterion(x, y)
            optimizer.zero_grad()  # reset your gradient
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(train_loader)
        train_loss_record.append(total_loss)

        # validation part
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x = model(x)
                x = x.squeeze()
                loss = criterion(x, y)
                total_loss += loss.item()

        total_loss /= len(valid_loader)
        valid_loss_record.append(total_loss)

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), 'model.ckpt')

    print(f'best = {best_loss}')

    import numpy as np
    from matplotlib import pyplot as plt
    plt.plot(np.arange(len(train_loss_record)), np.array(train_loss_record),label='train loss')
    plt.legend()
    plt.plot(np.arange(len(valid_loss_record)), np.array(valid_loss_record),label='valid loss')
    plt.legend()
    plt.savefig('img')
    plt.show()

if __name__ == '__main__':
    model = My_Model()
    train_loader = get_loader(train_path, mode='train', batch_size=2)
    valid_loader = get_loader(train_path, mode='valid', batch_size=2)
    train(train_loader=train_loader, valid_loader=valid_loader, model=model,
          total_epcohs=150)
