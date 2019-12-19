def unique(labels):
{
unique_list=[]
for x in labels:
  if x not in unique_list:
    unique_list.append(x)
count=len(unique_list)
return count
}

def one_hot_encode(labels):
{
cnt=unique(labels);
ohe_labels=[]
for x in labels:
  l=[]
  for i in range(0,cnt):
    l.append(0)
  l[x-1]=1
  ohe_labels.append(l)
return ohe_labels
}
