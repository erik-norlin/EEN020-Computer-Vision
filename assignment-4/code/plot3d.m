function plot3d(x, varargin)
    hold on
    plot3(x(1,:), x(2,:), x(3,:), '.', varargin{:})
    axis equal
end