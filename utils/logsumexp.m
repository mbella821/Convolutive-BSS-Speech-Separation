function s = logsumexp(x, dim)
% LOGSUMEXP Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
%   s = logsumexp(x, dim) returns the log of the sum of exponentials of x.
%   By default dim = 1 (columns).
%
%   This is a standard utility for stable computation of posterior
%   probabilities in the EM algorithm.

if nargin == 1
    % Determine which dimension sum will use
    dim = find(size(x)~=1,1);
    if isempty(dim), dim = 1; end
end

% Subtract the largest in each column (or row)
y = max(x,[],dim);
x = bsxfun(@minus,x,y);
s = y + log(sum(exp(x),dim));
i = find(~isfinite(y));
if ~isempty(i)
    s(i) = y(i);
end
end
