(* ::Package:: *)

mean[t_, n_] = (1 - (1 - Log[n] / t)^2)^(1 / 2) * t;
var[t_, n_] = (Log[n]^2 / t)^(2 / 3) * 2^(2 / 3) * (1 - Log[n] / t)^(4 / 3) * 0.8133 / (1 - (1 - Log[n] / t)^2);


arguments = $CommandLine;
nstr = Part[arguments,4];
nParticles = Read[StringToStream[nstr], Number];
maxDistance = Read[StringToStream[Part[arguments, 5]], Number]


ShortTimeAsym[xvals_, nParticles_] :=
    Module[{n = nParticles, x = xvals, time1, time2, t},
        time2[x_, n_] :=
            t /. FindRoot[Abs[mean[t, n] + Sqrt[var[t, n]] - x] == 0, {t, N[3.8 * 10^(-5) * x^3]}, MaxIterations -> 1000, AccuracyGoal -> Infinity, PrecisionGoal -> 16];
        time1[x_, n_] :=
            t /. FindRoot[Abs[mean[t, n] - Sqrt[var[t, n]] - x] == 0, {t, N[3.8 * 10^(-5) * x^3]}, MaxIterations -> 1000, AccuracyGoal -> Infinity, PrecisionGoal -> 16];
        Return[((time2[x, n] - time1[x, n]) / 2)^2, Module]
    ];


minDistance = IntegerPart[Log[nParticles]]+1


var = Table[ShortTimeAsym[x, nParticles], {x, minDistance, maxDistance}];


xvals = Range[minDistance, maxDistance];


Export["/home/jacob/Desktop/varianceShortTime.txt", var, "Table"];
Export["/home/jacob/Desktop/distances.txt", xvals, "Table"];


KPZTime = {25 / 100, 35 / 100, 50 / 100, 75 / 100, 12 / 10, 2, 35 / 10, 65 / 10, 13, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000};


KPZVar = {1.75713, 1.70908, 1.62923, 1.53633, 1.4361, 1.33806, 1.24323, 1.15268, 1.06801, 1.00314, 0.948845, 0.907622, 0.869637, 0.850573, 0.837569, 0.82681, 0.821864, 0.818774, 0.815189};


data = Transpose @ {KPZTime, KPZVar};


f = Interpolation[data, InterpolationOrder -> 1];


KPZFunc[t_] = Sqrt[Pi / 2] * (t / 2)^(-1 / 6) + (1 + 5 / 4 * Pi - 8 * Pi / 3 / Sqrt[3]) * (t / 2)^(1 / 3);


BetterInterpolation[time_] = Piecewise[{{N[KPZFunc[time]], time < 1 / 3}, {f[time], 1 / 3 <= time < 20000}, {0.815189, time >= 20000}}];


LongTimeVar[n_, time_] :=
    N[time / Log[n] / 2 * BetterInterpolation[N[4 (Log[n]^2) / time]] * 2^(-2 / 3) * (4 (Log[n]^2) / time)^(2 / 3)];


LongTimeAsym[xvals_, nParticles_] :=
    Module[{n = nParticles, x = xvals, time1, time2, t},
        time2[x_, n_] :=
            t /. FindRoot[Abs[mean[t, n] + Sqrt[LongTimeVar[n, t]] - x] == 0, {t, N[x^2 / 6 / (Log[n]^(3 / 4))], 2, Infinity}, MaxIterations -> 1000, AccuracyGoal -> Infinity, PrecisionGoal -> 16];
        time1[x_, n_] :=
            t /. FindRoot[Abs[mean[t, n] - Sqrt[LongTimeVar[n, t]] - x] == 0, {t, N[x^2 / 6 / (Log[n]^(3 / 4))], 2, Infinity}, MaxIterations -> 1000, AccuracyGoal -> Infinity, PrecisionGoal -> 16];
        Return[((time2[x, n] - time1[x, n]) / 2)^2, Module]
    ];


LongVar = Table[LongTimeAsym[x, nParticles], {x, minDistance, maxDistance}];


Export["/home/jacob/Desktop/varianceLongTime.txt", LongVar, "Table"];


mean[t_, n_] = (1 - (1 - Log[n] / t)^2)^(1 / 2) * t;
maxVar[t_, n_] = (Log[n]^2 / t)^(2 / 3) * 2^(2 / 3) * (1 - Log[n] / t)^(4 / 3) * 0.8133 / (1 - (1 - Log[n] / t)^2) + (t/Log[n] -1)^2 / (2*t/Log[n]-1) * Pi^2 / 6;


MaxAsym[xvals_, nParticles_] := 
	Module[{n = nParticles, x=xvals, time1, time2, t}, 
		time2[x_, n_] := t /. FindRoot[Abs[mean[t, n] + Sqrt[maxVar[t, n]] - x] == 0, {t, N[x^4 / 6 / (Log[n]^(3 / 4))], 2, Infinity}, MaxIterations -> 1000, AccuracyGoal -> Infinity, PrecisionGoal -> 16]; 
		time1[x_, n_] := t /. FindRoot[Abs[mean[t, n] - Sqrt[maxVar[t, n]] - x] == 0, {t, N[x^4 / 6 / (Log[n]^(3 / 4))], 2, Infinity}, MaxIterations -> 1000, AccuracyGoal -> Infinity, PrecisionGoal -> 16]; 
		Return[((time2[x, n] - time1[x, n]) / 2)^2, Module] 
	];


MaxVar = Table[MaxAsym[x, nParticles], {x, minDistance, maxDistance}];


Export["/home/jacob/Desktop/varianceMax.txt", MaxVar, "Table"];
