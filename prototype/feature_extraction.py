import json
import sqlite3
import torch

batch_size = 128
dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=2)
model_path = "./v3.pth"
net = PreLeNet(3)
net.load_state_dict(torch.load(model_path))
dbname = "features.db"
conn = sqlite3.connect(dbname)
c = conn.cursor()
c.execute("create table featuretable (id integer PRIMARY KEY, label integer, image text, feature text)")
sql = "insert into featuretable (id, label, image, feature) values (?, ?, ?, ?)"

d = {}
for i, (inputs, labels) in enumerate(dataloader):
    outputs, features = net(inputs)
    for label, image, feature in zip(labels, inputs, features):
        label = int(label)
        image = json.dumps(image.cpu().numpy().tolist())
        feature = json.dumps(feature.data.cpu().numpy().tolist())
        data = (id, label, image, feature)
        c.execute(sql, data)
        id += 1

conn.commit()
conn.close()
