function Make_Hackaton_Sim_Comb(Exp_Num,N_Users,False_Mass,False_Num,User_Removed_Frac,Comp_Removed_Frac,Use_Span)
N_Networks=3;
ll=0;
for ii=1:N_Networks
    for jj=ii+1:N_Networks
        ll=ll+1;
        Weight_User=10.^(rand(N_Users,1)*Use_Span); % this is the weight of each user
        %% put real and false assocations
        A_Real=eye(N_Users); %real matrix (diagonal)
        A=zeros(N_Users); %
        for i=1:N_Users
            A(i,i)=Weight_User(i); % put real assocations
            ztmp=Choose_Part(N_Users,rand(1)*False_Num); %put false associations
            for k=1:numel(ztmp)
                A(i,ztmp)=round(Weight_User(ceil(rand(1)*N_Users))*False_Mass*rand(1));
            end
        end
        
        % remove some users
        ztmp=Choose_Part(N_Users,User_Removed_Frac); A(:,ztmp)=0;A_Real(:,ztmp)=0;
        % Remove some computers
        ztmp=Choose_Part(N_Users,Comp_Removed_Frac);A(ztmp,:)=0;A_Real(ztmp,:)=0;
        
        % Write results as edge list.
        [x,y]=find(A>0);
        a=A(A>0);
        A=[x y a];
        
        [x,y]=find(A_Real>0);
        a=A_Real(A_Real>0);
        A_Real=[x y a];
        csvwrite(sprintf('Obs_Pair_K_Network_%d_Graph_%d.csv',Exp_Num,ll),A);
        csvwrite(sprintf('Real_Pair_K_Network_%d_Graph_%d.csv',Exp_Num,ll),A_Real);
    end
end
