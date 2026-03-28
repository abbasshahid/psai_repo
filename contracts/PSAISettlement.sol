// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title PSAISettlement
 * @notice Reference implementation of the PSAI on-chain settlement contract.
 *
 * Implements (from the provided PSAI system model):
 *  - Commit–reveal: com_t = H(enc(a_t) || nonce)  (Eq. 33)
 *  - Predictors: q_i(t)=sigma(w_q^T x_i(t)), rho_i(t)=sigma(w_rho^T x_i(t)) (Eqs. 7–8)
 *  - delta: δ_i(t)=exp(η^T x_i(t)) * q_i(t)^λ * (1-ρ_i(t))^λ  (Eq. 9)
 *  - Reward kernel (Eq. 26) and budget balance (Eq. 28)
 *  - Penalty (Eq. 27) with bounds
 *
 * IMPORTANT:
 * 1) On-chain computation of exp/sigmoid with high precision is expensive.
 *    This contract uses fixed-point approximations with bounded inputs.
 * 2) For production, consider off-chain computation with on-chain verification
 *    (e.g., SNARKs) or smaller committees.
 */
contract PSAISettlement {
    // -----------------------------
    // Types / Storage
    // -----------------------------
    struct Action {
        uint256 alpha;   // scaled by 1e6
        uint256 beta;    // scaled by 1e6
        uint256 lambda_; // scaled by 1e6
        uint256 kappa;   // scaled by 1e6 (not used in settlement; logged)
        int256[] eta;    // length K, scaled by 1e6
    }

    struct ValidatorInfo {
        address payout;
        uint256 stake;         // wei-like units
        int256[] x;            // length K, scaled by 1e6
        uint256 m;             // QoS score in [0,1e6]
        uint256 z;             // misbehavior indicator in [0,1e6]
        bool exists;
    }

    uint256 public constant SCALE = 1e6;
    uint256 public constant ONE = 1e6;

    // Bounds for actions (protocol parameters)
    uint256 public alphaMin = 0;
    uint256 public alphaMax = 5e6;   // up to 5
    uint256 public betaMin  = 0;
    uint256 public betaMax  = 5e6;   // up to 5
    uint256 public lambdaMax = 5e6;  // up to 5
    int256  public etaMaxAbs = 2e6;  // |eta_k| <= 2

    // Predictor weights (w_q, w_rho) stored on-chain for reproducibility
    int256[] public wq;   // length K, scaled by 1e6
    int256[] public wr;   // length K, scaled by 1e6

    // Epoch state
    uint256 public epoch;
    bytes32 public currentCommit;  // com_t
    bool public commitSet;

    mapping(uint256 => bytes32) public commitOfEpoch;
    mapping(uint256 => Action) public actionOfEpoch;

    address public owner;

    // Validators in current epoch
    address[] public validatorIds;
    mapping(address => ValidatorInfo) public validators;

    // Reentrancy guard
    uint256 private locked = 1;

    // -----------------------------
    // Events
    // -----------------------------
    event Committed(uint256 indexed epoch, bytes32 commitHash);
    event Revealed(uint256 indexed epoch, bytes32 commitHash);
    event Settled(uint256 indexed epoch, uint256 rewardPool, uint256 totalPaid, uint256 totalSlashed);
    event ValidatorUpdated(address indexed id, address payout, uint256 stake);
    event WeightsUpdated(uint256 K);

    modifier onlyOwner() { require(msg.sender == owner, "not owner"); _; }
    modifier nonReentrant() { require(locked == 1, "reentrant"); locked = 2; _; locked = 1; }

    constructor(int256[] memory _wq, int256[] memory _wr) {
        owner = msg.sender;
        require(_wq.length == _wr.length && _wq.length > 0, "K mismatch");
        wq = _wq;
        wr = _wr;
        emit WeightsUpdated(_wq.length);
    }

    // -----------------------------
    // Admin / Setup
    // -----------------------------
    function setBounds(uint256 _alphaMax, uint256 _betaMax, uint256 _lambdaMax, int256 _etaMaxAbs) external onlyOwner {
        alphaMax = _alphaMax;
        betaMax = _betaMax;
        lambdaMax = _lambdaMax;
        etaMaxAbs = _etaMaxAbs;
    }

    function upsertValidator(
        address id,
        address payout,
        uint256 stake,
        int256[] calldata x,
        uint256 m,
        uint256 z
    ) external onlyOwner {
        uint256 K = wq.length;
        require(x.length == K, "x length");
        require(m <= ONE && z <= ONE, "m/z range");

        ValidatorInfo storage v = validators[id];
        if (!v.exists) {
            validatorIds.push(id);
            v.exists = true;
        }
        v.payout = payout;
        v.stake = stake;
        v.m = m;
        v.z = z;

        // store x
        delete v.x;
        for (uint256 k = 0; k < K; k++) {
            v.x.push(x[k]);
        }
        emit ValidatorUpdated(id, payout, stake);
    }

    function clearValidators() external onlyOwner {
        for (uint256 i = 0; i < validatorIds.length; i++) {
            delete validators[validatorIds[i]];
        }
        delete validatorIds;
    }

    // -----------------------------
    // Commit–Reveal (Eq. 33)
    // -----------------------------
    function commitAction(bytes32 com) external onlyOwner {
        require(!commitSet, "commit already set");
        currentCommit = com;
        commitOfEpoch[epoch] = com;
        commitSet = true;
        emit Committed(epoch, com);
    }

    function revealAction(Action calldata a, bytes32 nonce) external onlyOwner {
        require(commitSet, "no commit");
        bytes32 com = keccak256(abi.encodePacked(encodeAction(a), nonce));
        require(com == currentCommit, "bad reveal");

        Action storage s = actionOfEpoch[epoch];
        s.alpha = clampU(a.alpha, alphaMin, alphaMax);
        s.beta = clampU(a.beta, betaMin, betaMax);
        s.lambda_ = clampU(a.lambda_, 0, lambdaMax);
        s.kappa = a.kappa;

        // clamp eta
        uint256 K = wq.length;
        require(a.eta.length == K, "eta length");
        delete s.eta;
        for (uint256 k = 0; k < K; k++) {
            int256 ek = a.eta[k];
            require(ek <= etaMaxAbs && ek >= -etaMaxAbs, "eta bound");
            s.eta.push(ek);
        }

        emit Revealed(epoch, currentCommit);

        // advance
        commitSet = false;
        currentCommit = bytes32(0);
    }

    function encodeAction(Action calldata a) public pure returns (bytes memory) {
        return abi.encode(a.alpha, a.beta, a.lambda_, a.kappa, a.eta);
    }

    // -----------------------------
    // Settlement (Algorithm 1)
    // -----------------------------
    /**
     * @notice Settle rewards and penalties for current epoch.
     * @dev For simplicity, this function emits computed totals and per-validator transfers
     *      are represented as events only. In a production contract, integrate token transfers,
     *      staking/slashing modules, and robust accounting.
     */
    function settle(uint256 rewardPool) external onlyOwner nonReentrant {
        Action storage a = actionOfEpoch[epoch];
        require(a.eta.length == wq.length, "no revealed action");

        // compute total stake
        uint256 N = validatorIds.length;
        require(N > 0, "no validators");
        uint256 totalStake = 0;
        for (uint256 i = 0; i < N; i++) totalStake += validators[validatorIds[i]].stake;
        require(totalStake > 0, "zero stake");

        // precompute weights w_i = exp(alpha*m_i) * delta_i
        uint256[] memory wi = new uint256[](N);
        uint256 Z = 0;

        for (uint256 i = 0; i < N; i++) {
            ValidatorInfo storage v = validators[validatorIds[i]];
            // q and rho via sigmoid of dot product
            int256 dq = dotScaled(wq, v.x);      // scaled by 1e6
            int256 dr = dotScaled(wr, v.x);      // scaled by 1e6
            uint256 q = sigmoid(dq);             // in [0,1e6]
            uint256 rho = sigmoid(dr);           // in [0,1e6]

            // delta = exp(eta^T x) * q^lambda * (1-rho)^lambda
            int256 de = dotScaled(a.eta, v.x);   // scaled
            uint256 exp1 = expApprox(de);        // scaled 1e6
            uint256 qPow = pow01(q, a.lambda_);  // scaled 1e6
            uint256 oneMinusRho = ONE - rho;
            uint256 rPow = pow01(oneMinusRho, a.lambda_);
            uint256 delta = mulScale(mulScale(exp1, qPow), rPow); // 1e6

            // exp(alpha*m)
            // alpha (1e6), m (1e6) -> alpha*m/1e6 in 1e6; pass as signed
            int256 am = int256((a.alpha * v.m) / SCALE);
            uint256 exp2 = expApprox(am); // 1e6

            uint256 w = mulScale(exp2, delta);
            wi[i] = w;
            Z += w;
        }
        require(Z > 0, "zero Z");

        uint256 totalPaid = 0;
        uint256 totalSlashed = 0;

        for (uint256 i = 0; i < N; i++) {
            ValidatorInfo storage v = validators[validatorIds[i]];
            // p_i = rewardPool * w_i / Z   (Eq. 26, budget-balanced)
            uint256 p = (rewardPool * wi[i]) / Z;
            totalPaid += p;

            // l_i = min{ stake, beta*stake*z*(1+lambda*rho) } (Eq. 27)
            int256 dr = dotScaled(wr, v.x);
            uint256 rho = sigmoid(dr); // recompute (could cache)
            uint256 onePlus = ONE + (a.lambda_ * rho) / SCALE; // 1e6
            uint256 l = (a.beta * v.stake) / SCALE;
            l = (l * v.z) / SCALE;
            l = (l * onePlus) / SCALE;
            if (l > v.stake) l = v.stake;
            totalSlashed += l;

            // NOTE: token transfer and slashing are protocol-specific.
            // Here we only emit totals (and users can verify deterministic computation).
        }

        emit Settled(epoch, rewardPool, totalPaid, totalSlashed);
        epoch += 1;
    }

    // -----------------------------
    // Fixed-point helpers
    // -----------------------------
    function clampU(uint256 x, uint256 lo, uint256 hi) internal pure returns (uint256) {
        if (x < lo) return lo;
        if (x > hi) return hi;
        return x;
    }

    function mulScale(uint256 a, uint256 b) internal pure returns (uint256) {
        return (a * b) / SCALE;
    }

    function dotScaled(int256[] storage a, int256[] storage b) internal view returns (int256) {
        require(a.length == b.length, "dot len");
        int256 acc = 0;
        for (uint256 i = 0; i < a.length; i++) {
            // (a_i * b_i) / 1e6 keeps scaling stable
            acc += (a[i] * b[i]) / int256(SCALE);
        }
        return acc; // scaled by 1e6
    }

    /**
     * @dev Sigmoid approximation on scaled input x (1e6). Output in [0,1e6].
     * Uses: sigmoid(x) ≈ 1 / (1 + exp(-x)).
     * We compute exp(-x) via expApprox and then divide.
     */
    function sigmoid(int256 x) internal view returns (uint256) {
        // clamp to avoid overflow and extreme costs
        if (x > int256(8e6)) return ONE;      // ~1
        if (x < -int256(8e6)) return 0;       // ~0
        uint256 ex = expApprox(-x);           // exp(-x) scaled 1e6
        // 1 / (1+ex) in fixed point:
        // ONE / (ONE + ex) * ONE
        return (ONE * SCALE) / (SCALE + ex);  // equals 1e6/(1+exp(-x))
    }

    /**
     * @dev exp approximation on scaled input x (1e6), returns exp(x/1e6) scaled 1e6.
     * Uses a 5th-order Taylor around 0 with input clamping.
     */
    function expApprox(int256 x) internal pure returns (uint256) {
        // clamp x to [-6, +6] to bound approximation error and overflow
        if (x > int256(6e6)) x = int256(6e6);
        if (x < -int256(6e6)) x = -int256(6e6);

        // Convert to y = x / 1e6 in fixed point 1e6
        int256 y = x;

        // Compute exp(y) ~ 1 + y + y^2/2 + y^3/6 + y^4/24 + y^5/120
        // All in SCALE.
        int256 term1 = int256(SCALE);
        int256 term2 = y;
        int256 y2 = (y * y) / int256(SCALE);
        int256 y3 = (y2 * y) / int256(SCALE);
        int256 y4 = (y3 * y) / int256(SCALE);
        int256 y5 = (y4 * y) / int256(SCALE);

        int256 res = term1
            + term2
            + (y2 / 2)
            + (y3 / 6)
            + (y4 / 24)
            + (y5 / 120);

        if (res < 0) return 0;
        return uint256(res);
    }

    /**
     * @dev Compute (x/1e6)^(lambda/1e6) where x in [0,1e6] and lambda in [0,5e6].
     * Uses exp(lambda * ln(x)) approximation with coarse ln via series; bounded inputs.
     * For simplicity, we approximate using:
     *  pow01(x,lambda) ≈ exp( lambda * ln(max(x,1)) ).
     */
    function pow01(uint256 x, uint256 lambda_) internal pure returns (uint256) {
        if (x == 0) return 0;
        if (x == ONE) return ONE;
        // clamp
        if (lambda_ == 0) return ONE;

        // ln(x) where x in (0,1): ln(x) = -ln(1/(x)).
        // Approximate ln(x) using ln around 1: ln(1+u) ~ u - u^2/2 + u^3/3 with u=(x-1).
        // Represent x as fixed point in [0,1]. Convert to u = (x-ONE)/ONE.
        int256 u = int256(x) - int256(ONE); // in [-1e6,0)
        // ln ≈ u - u^2/2 + u^3/3, scaled 1e6
        int256 u2 = (u * u) / int256(SCALE);
        int256 u3 = (u2 * u) / int256(SCALE);
        int256 ln = u - (u2 / 2) + (u3 / 3);

        // y = lambda * ln / 1e6  (scaled 1e6)
        int256 y = (int256(lambda_) * ln) / int256(SCALE);
        return expApprox(y);
    }
}
