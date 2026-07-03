# Jason Peng's clv-dashboard 🐉 恭喜发财

Customer Lifetime Value modeling with BG/NBD + Gamma-Gamma

A web dashboard that predicts how much each of your customers is worth in
the future, using two well-established statistical models from marketing
science. Upload transaction history, get back a ranked list of customers
with their predicted future value.


What this actually does

BG/NBD (Beta Geometric / Negative Binomial Distribution)
Looks at when and how often each customer has purchased, and predicts:


How many purchases they'll make in a future period
The probability they're still an active customer (vs. having silently
churned — there's no "cancel" button in retail, so this has to be inferred)


Gamma-Gamma
Looks at how much customers spend per order, and predicts their expected
average future order value. Deliberately kept separate from BG/NBD because
purchase frequency and order size are assumed to be independent.

Combined: expected future purchases × expected order value × a discount
rate = predicted Customer Lifetime Value (CLV) for every customer.

These models are built for non-contractual, repeat-purchase businesses
(ecommerce, retail, restaurants) — not subscriptions, where churn is
explicit and better modeled differently.
