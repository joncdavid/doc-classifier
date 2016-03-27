#!/usr/bin/env octave -qf
# filename: build_beta_plot.m
# authors: Jon David and Jarrett Decker
# description:
#   Uses GNU Octave to plot beta-value vs model-accuracy.
D = load("./model_results/beta_vs_acc.data");
x = D(:,1);
y = D(:,2);
plot(x,y);
semilogx(x,y);
xlabel("log scale of beta-values in range (0,1.0]")
ylabel("accuracy")
title("Naive Bayes Classifier Accuracy")
print -djpg "./model_results/beta_vs_acc.png"

