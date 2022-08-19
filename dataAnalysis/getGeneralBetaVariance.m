(* ::Package:: *)

II[\[Theta]_, \[Alpha]_, \[Beta]_] = (PolyGamma[1, \[Theta] + \[Alpha] + \[Beta]] - PolyGamma[1, \[Theta] + \[Alpha]]) / (PolyGamma[1, \[Theta]] - PolyGamma[1, \[Theta] + \[Alpha] + \[Beta]]) (PolyGamma[\[Theta] + \[Alpha] + \[Beta]] - PolyGamma[\[Theta]]) + PolyGamma[\[Theta] + \[Alpha] + \[Beta]] - PolyGamma[\[Theta] + \[Alpha]];
\[Sigma][\[Theta]_, \[Alpha]_, \[Beta]_] = (1 / 2 * (PolyGamma[2, \[Theta] + \[Alpha]] - PolyGamma[2, \[Theta] + \[Alpha] + \[Beta]] + (PolyGamma[1, \[Alpha] + \[Theta]] - PolyGamma[1, \[Alpha] + \[Beta] + \[Theta]]) / (PolyGamma[1, \[Theta]] - PolyGamma[1, \[Alpha] + \[Beta] + \[Theta]]) * (PolyGamma[2, \[Theta] + \[Alpha] + \[Beta]] - PolyGamma[2, \[Theta]])))^(1 / 3);
x[\[Theta]_, \[Alpha]_, \[Beta]_] = (PolyGamma[1, \[Theta] + \[Alpha] + \[Beta]] + PolyGamma[1, \[Theta]] - 2 * PolyGamma[1, \[Theta] + \[Alpha]]) / (PolyGamma[1, \[Theta]] - PolyGamma[1, \[Theta] + \[Alpha] + \[Beta]]);


arguments = $CommandLine;
nParticles = Read[StringToStream[Part[arguments,4]], Number];
maxDistance = Read[StringToStream[Part[arguments, 5]], Number]; 
beta = Read[StringToStream[Part[arguments, 6]], Number];


minDistance = IntegerPart[Log[nParticles]]+1


fShortAsym[nParticles_, time_, betaval_] := 
Module[{n=nParticles, t=time,beta=betaval, theta0, theta0vals, varChi=0.8133, var, dervI,dervV, \[Theta]}, 
dervI[\[Theta]_, \[Alpha]_, \[Beta]_] = N[D[II[\[Theta],beta, beta], \[Theta]]]; 
dervV[\[Theta]_, \[Alpha]_, \[Beta]_] =  N[D[x[\[Theta],beta, beta], \[Theta]]];
theta0[t_] := \[Theta] /. FindRoot[II[\[Theta],beta,beta]== Log[n]/t, {\[Theta], 0.001, 0, 500}];
theta0vals = Map[theta0, t];
var=varChi*t^(2/3) * (\[Sigma][theta0vals, beta, beta]*dervV[theta0vals, beta, beta] / (dervI[theta0vals, beta, beta]))^2;
Return[var, Module]] ;


t = Range[minDistance, maxDistance];
var = fShortAsym[nParticles, t, 0.1];
stringfile = "/home/jacob/Desktop/variance" <> ToString[beta] <> ".txt"

Export[
    "/home/jacob/Desktop/variance" <> ToString[beta] <> ".txt", var, "Table"
]; Export[
    "/home/jacob/Desktop/times" <> ToString[beta] <> ".txt", t
];
