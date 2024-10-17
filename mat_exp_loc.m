function B = mat_exp_loc(A,lambda)

B = expm(lambda*A);

D = diag(B);
invD = sqrt(D).^(-1);
invDmat = diag(invD);
B = invDmat*B*invDmat;

end

