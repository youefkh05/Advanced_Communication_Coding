function [res1,res2,size1,size2]=splitf(inf)


%inf=[0.25;0.25]
for t=1:1:length(inf)
    a=sum(inf(1:t));
    b=sum(inf(t+1:end));
    c(t)=(a-b);
end

c=abs(c);
[y,t]=min(c);
res1=inf(1:t);
size1=t;
res2=inf(t+1:length(inf));
size2=length(inf)-t;