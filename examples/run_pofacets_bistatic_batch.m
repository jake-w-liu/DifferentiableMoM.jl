function run_pofacets_bistatic_batch(input_mat, out_csv, out_summary_csv, freq_ghz, varargin)
% Headless POFacets bistatic run on an existing .mat CAD model.
% Uses POFacets facetRCS and CalcBistat-equivalent settings (no GUI).
%
% Arguments:
%   input_mat        : path to airplane.mat
%   out_csv          : output CSV for full spherical grid
%   out_summary_csv  : output CSV for phi~=0 cut summary
%   freq_ghz         : frequency in GHz (default 3.0)
% Optional name/value args:
%   'itheta_deg'     : incidence theta in degrees (default 180)
%   'iphi_deg'       : incidence phi in degrees (default 0)
%   'i_pol'          : 1 => theta-pol, 2 => phi-pol (default 1)
%   'iflag'          : 0 => use illumination test, 1 => disable (default 0)
%   'flip_winding'   : true => swap facet columns 2<->3 before solve (default false)

if nargin < 4
    freq_ghz = 3.0;
end
if nargin < 3
    error('Usage: run_pofacets_bistatic_batch(input_mat, out_csv, out_summary_csv, freq_ghz)');
end

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
pof_dir = fullfile(this_dir, '..', '..', 'pofacets4.5', 'pofacets4.5');
addpath(pof_dir);

S = load(input_mat);
coord = double(S.coord);
facet = double(S.facet);

% Matched to Julia ex_obj_rcs_pipeline settings.
Ntheta = 121;
Nphi = 36;
itheta_deg = 180.0;  % incidence propagation along -z
iphi_deg = 0.0;
i_pol = 1;           % Et=1, Ep=0
rsmethod = 1;        % use Rs from facet (airplane.mat has Rs=0 -> PEC)
iflag = 0;
Lt = 1e-5;
Nt = 5;
corr = 0.0;
stdv = 0.0;
flip_winding = false;

for k = 1:2:numel(varargin)
    name = lower(varargin{k});
    val = varargin{k+1};
    switch name
        case 'itheta_deg'
            itheta_deg = double(val);
        case 'iphi_deg'
            iphi_deg = double(val);
        case 'i_pol'
            i_pol = double(val);
        case 'iflag'
            iflag = double(val);
        case 'flip_winding'
            flip_winding = logical(val);
        otherwise
            error('Unknown option: %s', varargin{k});
    end
end

C0 = 3e8;
wave = C0 / (freq_ghz * 1e9);
bk = 2*pi / wave;
corel = corr / wave;
delsq = stdv^2;
cfac1 = exp(-4*bk^2*delsq);
cfac2 = 4*pi*(bk*corel)^2*delsq;

node1 = facet(:,1);
node2 = facet(:,2);
node3 = facet(:,3);
if flip_winding
    tmp = node2;
    node2 = node3;
    node3 = tmp;
end
ilum = facet(:,4);
Rs = facet(:,5);

ntria = size(facet,1);
nverts = size(coord,1);
vind = [node1 node2 node3];

x = coord(:,1);
y = coord(:,2);
z = coord(:,3);
r = [x y z];

N = zeros(ntria,3);
Area = zeros(ntria,1);
alpha = zeros(ntria,1);
beta = zeros(ntria,1);

for i = 1:ntria
    A = r(vind(i,2),:) - r(vind(i,1),:);
    B = r(vind(i,3),:) - r(vind(i,2),:);
    C = r(vind(i,1),:) - r(vind(i,3),:);
    nvec = -cross(B,A);
    d1 = norm(A); d2 = norm(B); d3 = norm(C);
    ss = 0.5 * (d1 + d2 + d3);
    Area(i) = sqrt(max(ss*(ss-d1)*(ss-d2)*(ss-d3), 0));
    Nn = norm(nvec);
    if Nn == 0
        N(i,:) = [0 0 1];
    else
        N(i,:) = nvec / Nn;
    end
    beta(i) = acos(max(min(N(i,3),1),-1));
    alpha(i) = atan2(N(i,2),N(i,1));
end

rad = pi/180;
ithetar = itheta_deg * rad;
iphir = iphi_deg * rad;

if i_pol == 1
    Et = 1 + 1i*0;
    Ep = 0 + 1i*0;
else
    Et = 0 + 1i*0;
    Ep = 1 + 1i*0;
end

cpi = cos(iphir); spi = sin(iphir);
sti = sin(ithetar); cti = cos(ithetar);
uui = cti*cpi; vvi = cti*spi; wwi = -sti;
e0 = [uui*Et - spi*Ep, vvi*Et + cpi*Ep, wwi*Et];

theta_vec = ((1:Ntheta) - 0.5) * pi / Ntheta;
phi_vec = ((1:Nphi) - 0.5) * 2*pi / Nphi;

Nobs = Ntheta * Nphi;
theta_deg = zeros(Nobs,1);
phi_deg = zeros(Nobs,1);
sigma_theta_db = zeros(Nobs,1);
sigma_phi_db = zeros(Nobs,1);
sigma_total_m2 = zeros(Nobs,1);
sigma_total_db = zeros(Nobs,1);

idx = 0;
for ip = 1:Nphi
    phr = phi_vec(ip);
    for it = 1:Ntheta
        idx = idx + 1;
        thr = theta_vec(it);

        sumt = 0 + 1i*0;
        sump = 0 + 1i*0;
        sumdt = 0;
        sumdp = 0;
        RCpar = 0;
        RCperp = 0;

        for m = 1:ntria
            [Ets, Etd, Eps, Epd] = facetRCS(thr, phr, ithetar, iphir, ...
                N(m,:), ilum(m), iflag, alpha(m), beta(m), Rs(m), Area(m), ...
                x, y, z, vind(m,:), e0, Nt, Lt, cfac2, corel, wave, ...
                0, 0, 0, rsmethod, RCpar, RCperp);
            sumt = sumt + Ets;
            sump = sump + Eps;
            sumdt = sumdt + abs(Etd);
            sumdp = sumdp + abs(Epd);
        end

        sth_db = 10*log10(4*pi*cfac1*(abs(sumt)^2 + sqrt(1-cfac1^2)*sumdt)/wave^2 + 1e-10);
        sph_db = 10*log10(4*pi*cfac1*(abs(sump)^2 + sqrt(1-cfac1^2)*sumdp)/wave^2 + 1e-10);

        sig_t = 4*pi * (abs(sumt)^2) / wave^2;
        sig_p = 4*pi * (abs(sump)^2) / wave^2;
        sig_tot = sig_t + sig_p;

        theta_deg(idx) = thr / rad;
        phi_deg(idx) = phr / rad;
        sigma_theta_db(idx) = sth_db;
        sigma_phi_db(idx) = sph_db;
        sigma_total_m2(idx) = sig_tot;
        sigma_total_db(idx) = 10*log10(max(sig_tot, 1e-30));
    end
end

T = table(theta_deg, phi_deg, sigma_theta_db, sigma_phi_db, sigma_total_m2, sigma_total_db);
out_dir = fileparts(out_csv);
if ~isempty(out_dir)
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
end
writetable(T, out_csv);

phi_abs = abs(phi_deg);
[~, i0] = min(phi_abs);
phi_target = phi_deg(i0);
cut_idx = abs(phi_deg - phi_target) < 1e-12;
cut_theta = theta_deg(cut_idx);
cut_sig = sigma_total_m2(cut_idx);
[cut_theta, ord] = sort(cut_theta);
cut_sig = cut_sig(ord);

Tcut = table(cut_theta, cut_sig, 10*log10(max(cut_sig,1e-30)), repmat(phi_target, size(cut_theta,1), 1), ...
    'VariableNames', {'theta_deg','sigma_po_m2','sigma_po_dBsm','phi_cut_deg'});
out_dir2 = fileparts(out_summary_csv);
if ~isempty(out_dir2)
    if ~exist(out_dir2, 'dir')
        mkdir(out_dir2);
    end
end
writetable(Tcut, out_summary_csv);

fprintf('Saved PO grid CSV: %s\n', out_csv);
fprintf('Saved PO phi-cut CSV: %s\n', out_summary_csv);
end
