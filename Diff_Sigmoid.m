function Return = Diff_Sigmoid(Vector)
Return = Sigmoid(Vector).*(1-Sigmoid(Vector));
end
