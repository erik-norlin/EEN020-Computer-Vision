function plot2d(x, varargin)
    hold on
    scatter(x(1,:), x(2,:), varargin{:})
end