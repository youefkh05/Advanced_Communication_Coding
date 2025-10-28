function res=two2o(intwo)

temp=num2str(intwo);
count=length(temp);
for i=1:1:count
    if(temp(i)=='2')
         res(i)='0';
    else
        res(i)='1';
    end
end

res(count+1)='-';