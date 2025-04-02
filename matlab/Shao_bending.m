% Full Phase Field Model with Reaction-Diffusion Coupling

% Define the full output path
output_dir = 'C:/Users/zackk/OneDrive/Desktop/Research/With bending results';
save_interval = round(0.5 / dt); 

% Make sure the directory exists
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Parameters from the table
Nx = 600; Ny = 200;
dx = 0.1; dy = 0.1;
dt = 1e-4/8
nSteps = 48000000;

% Physical parameters
gamma = 1.0;            % pN
kappa = 1.0;            % pN*um^2
alpha = 0.3;            % pN/um
beta = 0.5;             % pN/um
Ma = 4.0;               % pN/um^3
A0 = 50.24;             % um^2
epsilon = 1.0;          % um
tau = 2.62;             % pN*s/um^2

% Reaction-Diffusion constants
a = 0.084;              % s^-1
b = 1.146;              % s^-1
c = 0.0764;             % s^-1
e = 0.107;              % s^-1
DV = 0.382;             % um^2/s
DW = 0.0764;            % um^2/s

% Spatial grid
x = linspace(0, (Nx-1)*dx, Nx);
y = linspace(0, (Ny-1)*dy, Ny);
[X, Y] = meshgrid(x, y);

% Initial phase field: circle of radius 4 um
phi = zeros(Ny, Nx);
cx = Nx/2; cy = Ny/2;
radius = 4.0;
phi(((X - x(cx)).^2 + (Y - y(cy)).^2) < radius^2) = 1;
phi_prev=phi;
phi = phi + 0.01 * randn(size(phi));
phi(phi < 0) = 0;
phi(phi > 1) = 1;


% Initialize V and W inside the cell only
V = 1.1 * phi;

W = zeros(Ny, Nx);  % (200 x 600)
W0 = 10;

% Create the rear-loaded bundle shape
rear_shape = W0 * exp(-((Y - y(cy) + 5).^2) / (2 * 3^2));

% Apply to rear side of the cell only
W = rear_shape .* phi .* (Y < y(cy));
%GPU arrays:
phi = gpuArray(phi);
V = gpuArray(V);
W = gpuArray(W);
X = gpuArray(X);
Y = gpuArray(Y);
phi_prev = gpuArray(phi_prev);

% Custom Laplacian function
function lap = laplacian9(phi, dx, dy)
    %kernel = gpuArray([1 4 1; 4 -20 4; 1 4 1]) / 6;
    kernel =[1 4 1; 4 -20 4; 1 4 1] / 6;
    lap = conv2(phi, kernel, 'same') / dx^2;
end



G_prime = @(phi) 36 * phi .* (1 - phi) .* (1 - 2 * phi);
G_double_prime = @(phi) 36 * ((1 - phi) .* (1 - 2*phi) - phi .* (1 - 2*phi) - 2 * phi .* (1 - phi));

% Tracking center of mass and area
xcom = zeros(1, nSteps);
ycom = zeros(1, nSteps);
area_vals = zeros(1, nSteps);

for step = 1:nSteps
    % Tension
    lap_phi = laplacian9(phi, dx, dy);
    tension = gamma * (lap_phi - G_prime(phi) / epsilon^2);

    % Volume conservation
    A = sum(phi(:)) * dx * dy;
    phi_smooth = imgaussfilt(phi, 1);
    [dphix, dphiy] = gradient(phi_smooth, dx, dy);
    grad_phi_mag = sqrt(dphix.^2 + dphiy.^2);
    volume = -Ma * (A - A0) * grad_phi_mag;

    % Bending (now with FFT-based biharmonic)
    biharm_phi = laplacian9(lap_phi, dx, dy);
    lap_Gp = laplacian9(G_prime(phi), dx, dy);
    Gpp = G_double_prime(phi);
    bending = -kappa * (biharm_phi - lap_Gp - Gpp .* lap_phi + Gpp .* G_prime(phi));

    % Reaction-Diffusion term
    reaction_diffusion = (alpha * V - beta * W) .* grad_phi_mag;

    % Update phi
    dphi_dt = (tension + volume + bending + reaction_diffusion) / tau;
    phi = phi + dt * dphi_dt;
    phi(phi < 0) = 0;
    phi(phi > 1) = 1;


    % Update V and W
    RV = phi .* (a - b * V .* W.^2 - c * V);
    RW = phi .* (b * V .* W.^2 - e * W);

    [phix, phiy] = gradient(phi, dx, dy);
    [Vx, Vy] = gradient(V, dx, dy);
    [Wx, Wy] = gradient(W, dx, dy);
    
    phiVx = phix .* V + phi .* Vx;
    phiVy = phiy .* V + phi .* Vy;
    phiWx = phix .* W + phi .* Wx;
    phiWy = phiy .* W + phi .* Wy;
    
    div_phi_gradV = divergence(phiVx, phiVy);
    div_phi_gradW = divergence(phiWx, phiWy);

    V = V + dt ./ (phi + 1e-8) .* (RV + DV * div_phi_gradV);
    W = W + dt ./ (phi + 1e-8) .* (RW + DW * div_phi_gradW);

    V(phi < 1e-3) = 0;
    W(phi < 1e-3) = 0;

    % Track center of mass and area
    total_phi = sum(phi(:));
    xcom(step) = sum(sum(phi .* X)) / total_phi;
    ycom(step) = sum(sum(phi .* Y)) / total_phi;
    area_vals(step) = A;
    % if mod(step, 5000) == 0
    % fprintf("RD max: %.4f | mean: %.4f\n Area value: %.4f\n", max(reaction_diffusion(:)), mean(reaction_diffusion(:)),A);
    % end
    % if mod(step, 10000) == 0
    % delta_phi = abs(phi - phi_prev);
    % fprintf("Mean Δφ: %.6f | Max Δφ: %.6f\n", mean(delta_phi(:)), max(delta_phi(:)));
    % phi_prev = phi;
    % phi_mask = phi > 0.5;
    % [y_indices, x_indices] = find(phi_mask);
    % width = range(x_indices) * dx;
    % height = range(y_indices) * dy;
    % fprintf("Shape | width: %.2f µm, height: %.2f µm\n", width, height);
    % elongation = height / width;
    % fprintf("Elongation ratio: %.3f\n", elongation);
    % end
    %40000 
 if mod(step, save_interval) == 0
    fig = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
    t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    imagesc(phi, [0 1]);
    title('\phi'); axis equal tight; colorbar;

    nexttile;
    imagesc(V);
    title('V'); axis equal tight; colorbar;

    nexttile;
    imagesc(W);
    title('W'); axis equal tight; colorbar;

    sim_time = step * dt;
    filename = sprintf('cell_frame_step%07d_t%.2f.pdf', step, sim_time);
    full_path = fullfile(output_dir, filename);
    exportgraphics(t, full_path, 'ContentType', 'vector');
    close(fig);
end

end
