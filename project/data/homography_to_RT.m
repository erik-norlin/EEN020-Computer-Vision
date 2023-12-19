% -*- Octave -*-
function [R1, t1, R2, t2] = homography_to_RT(H, x1, x2)
  function [ra,rb] = unitize(a,b)
    denom = 1.0/sqrt(a^2+b^2);
    ra = a * denom; rb = b * denom;
  end

  #if det(H) < 0  H *= -1;  end
  # Check the right sign for H
  N = size(x1, 2);
  if size(x1, 1) != 3   x1 = [x1; ones(1, N)]; end
  if size(x2, 1) != 3   x2 = [x2; ones(1, N)]; end
  positives = sum((sum(x2 .* (H*x1), 1)) > 0);
  if positives < N/2   H *= -1; end

  [U,S,V] = svd(H);
  s1 = S(1,1)/S(2,2);
  s3 = S(3,3)/S(2,2);
  zeta = s1-s3;
  a1 = sqrt(1-s3^2);
  b1 = sqrt(s1^2-1);
  [a,b] = unitize(a1,b1);
  [c,d] = unitize( 1+s1*s3, a1*b1 );
  [e,f] = unitize( -b/s1, -a/s3 );
  v1 = V(:,1); v3 = V(:,3);
  n1 = b*v1-a*v3;
  n2 = b*v1+a*v3;
  R1 = U*[c,0,d; 0,1,0; -d,0,c]*V';
  R2 = U*[c,0,-d; 0,1,0; d,0,c]*V';
  t1 = e*v1+f*v3;
  t2 = e*v1-f*v3;
  if (n1(3)<0) t1 = -t1; n1 = -n1; end;
  if (n2(3)<0) t2 = -t2; n2 = -n2; end;

  % Move from Triggs' convention H = R*(I - t*n') to H&Z notation H = R - t*n'
  t1 = R1*t1;
  t2 = R2*t2;

  %c1 = -R1'*t1; c2 = -R2'*t2;
  %[(R1*n1)(3), (R2*n2)(3)]
  %[n1'*c1 + zeta, n2'*c2 + zeta]

  % Verify that we obtain the initial homography back
  %H /= S(2,2);
  %[norm(R1 - zeta*t1*n1' - H), norm(R2 - zeta*t2*n2' - H)]
end;
