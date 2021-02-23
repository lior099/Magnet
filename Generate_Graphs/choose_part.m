function ztmp=Choose_Part(N_Users,Frac)
ztmp=randperm(N_Users);
Frac
ztmp=ztmp(1:ceil(N_Users*Frac));
end

