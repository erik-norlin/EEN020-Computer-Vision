function y = pflat(x)
% Normalizes (m,n)-array x of n homogeneous points
% to last coordinate 1.
y = x ./ x(end,:);
end

