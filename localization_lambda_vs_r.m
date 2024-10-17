nr = 10;
lambda = zeros(nr,1);
B = zeros(nr,N,N);
l0 = 0.5;
l1 = 0.7;
eps = 0.1;
for num_neigh = 1:nr
    [B(num_neigh,:,:),lambda(num_neigh)] = ring_loc_param(N,num_neigh,eps,l0,l1,1e-6);
    l0 = l1;
    l1 = lambda(num_neigh);
end

%%
figure
plot(1:nr,lambda,'-o')
% hold on
% end
% hold off
set(gca,'FontSize',14)
xlabel('$r$','Interpreter','latex','FontSize',18)
ylabel('$\lambda$','Interpreter','latex','FontSize',18)
% legend()

%%
figure
plot(1:nr,1./lambda,'-o')
set(gca,'FontSize',14)
xlabel('$r$','Interpreter','latex','FontSize',18)
ylabel('$1/\lambda$','Interpreter','latex','FontSize',18)