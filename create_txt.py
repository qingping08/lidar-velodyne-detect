import os
f=open("test.txt",'w')
for i in range(1,447):
    s=str(i).rjust(10,'0')
    f.write(s)
    if i !=446:
     f.write('\n')
f.close()