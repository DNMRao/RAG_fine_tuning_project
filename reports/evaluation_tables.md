# Evaluation Tables

Retrieval on supported cases (k=3): recall@k = 1.0, MRR = 1.0

| Metric | Baseline RAG | Hybrid RAG + LoRA |
|---|---|---|
| decision_accuracy | 0.286 | 0.857 |
| term_coverage | 0.55 | 0.85 |
| citation_faithfulness | 0.0 | 1.0 |
| style_compliance | 0.0 | 1.0 |
| forbidden_claims | 0 | 0 |
| mean_latency_seconds | 1.316 | 1.484 |
| answered | 0 | 12 |
| abstained | 3 | 2 |
| rejected | 11 | 0 |

## Per Case

| Case | Expected | Top score | Recall@k | Baseline (coverage) | Hybrid (coverage) |
|---|---|---|---|---|---|
| eval-refund-ticket | answer | 0.743 | 1.0 | rejected (0.0) | answered (1.0) |
| eval-duplicate-payment | answer | 0.525 | 1.0 | rejected (1.0) | answered (1.0) |
| eval-authenticator | answer | 0.607 | 1.0 | rejected (0.0) | answered (0.5) |
| eval-passport | answer | 0.606 | 1.0 | rejected (0.0) | answered (1.0) |
| eval-packed-address | answer | 0.841 | 1.0 | rejected (0.0) | answered (0.0) |
| eval-standard-shipping | answer | 0.654 | 1.0 | rejected (1.0) | answered (1.0) |
| eval-annual-refund | answer | 0.705 | 1.0 | rejected (1.0) | answered (1.0) |
| eval-pause | answer | 0.891 | 1.0 | rejected (1.0) | answered (1.0) |
| eval-outage-updates | answer | 0.708 | 1.0 | rejected (1.0) | answered (1.0) |
| eval-log-secrets | answer | 0.654 | 1.0 | rejected (0.5) | answered (1.0) |
| eval-bank-chargeback | abstain | 0.662 | - | abstained (-) | answered (-) |
| eval-shipping-compensation | abstain | 0.549 | - | rejected (-) | answered (-) |
| eval-cafeteria | abstain | 0.132 | - | abstained (-) | abstained (-) |
| eval-bicycle | abstain | 0.208 | - | abstained (-) | abstained (-) |

## Model Outputs

**eval-refund-ticket**: A digital workshop starts in 12 hours. Can the customer still receive a refund?
- baseline — rejected (answer has no citation): Yes, the customer can still receive a refund. The context states that the digital workshop starts in 12 hours, and the customer provides the order ID and payment references within 30 calendar days. This means that the customer can receive a refund within 30 days of the workshop starting.
- hybrid — answered (citations match retrieved evidence): The digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window. [refunds]

**eval-duplicate-payment**: What must support collect for a duplicate payment investigation?
- baseline — rejected (answer has no citation): To collect for a duplicate payment investigation, the following must be fulfilled:

1. Collect the invoice number and both payment references.
2. Collect the order ID.
3. Collect the order ID and payment references.
4. Collect the order ID and payment references.
5. Collect the order ID and payment references.
6. Collect the order ID and payment references.
7
- hybrid — answered (citations match retrieved evidence): Please collect the invoice number and both payment references. Finance reviews a complete duplicate-payment request within 30 calendar days when the customer provides the order ID. [refunds]

**eval-authenticator**: A customer lost their authenticator. What information and process should support use?
- baseline — rejected (answer has no citation): A customer lost their authenticator. The customer should contact support to inform them of the loss and request a secure recovery link.
- hybrid — answered (citations match retrieved evidence): The authenticator should be reset using the secure upload portal, never ordinary chat. [account-security]

**eval-passport**: Should a customer paste a passport scan into chat to recover an account?
- baseline — rejected (answer has no citation): Based on the context, it is not recommended to paste a passport scan into chat to recover an account. The context does not provide any information about the customer's account, and the password is not provided. Therefore, the answer is no.
- hybrid — answered (citations match retrieved evidence): Paste the passport scan in the secure upload portal, never ordinary chat. [incidents]

**eval-packed-address**: Can support change a delivery address after the warehouse packed the order?
- baseline — rejected (answer has no citation): Yes, support can change a delivery address after the warehouse packs the order.
- hybrid — answered (citations match retrieved evidence): Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days. [shipping]

**eval-standard-shipping**: What is the usual standard shipping time?
- baseline — rejected (answer has no citation): The standard shipping time is three to five business days.
- hybrid — answered (citations match retrieved evidence): Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days. [shipping]

**eval-annual-refund**: Can an annual subscription receive a partial refund after 45 days?
- baseline — rejected (answer has no citation): Based on the context, it is not possible for an annual subscription to receive a partial refund after 45 days. The context states that the subscription is not fully refunded after the 30-day refund window, and the refunds are not returned to the original payment method.
- hybrid — answered (citations match retrieved evidence): No, the annual purchase does not renew the refund. Check the 30-day window before the purchase to ensure the cancellation is within the allowed time frame. [subscriptions]

**eval-pause**: How often and for how long can an active subscription be paused?
- baseline — rejected (answer has no citation): The context states that an active subscription can be paused once per calendar quarter for up to 30 days.
- hybrid — answered (citations match retrieved evidence): The active purchase may be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer. [subscriptions]

**eval-outage-updates**: How frequently should updates be published during a confirmed outage?
- baseline — rejected (answer has no citation): To determine the frequency of updates during a confirmed outage, we need to consider the time frame of the outage and the frequency of updates.

During a confirmed outage, the frequency of updates can vary depending on the specific outage and the customer's preferences. However, in general, it is recommended to publish updates at least every 30 minutes until the outage is resolved. This allows customers to receive
- hybrid — answered (citations match retrieved evidence): During a confirmed service outage, publish status updates at least every 30 minutes until recovery. Direct customers to the public status page for the latest confirmed information. [incidents]

**eval-log-secrets**: May diagnostic logs sent to support contain authentication tokens?
- baseline — rejected (answer has no citation): Yes, diagnostic logs may contain authentication tokens.
- hybrid — answered (citations match retrieved evidence): Yes, diagnostic logs may include request IDs and timestamps, but must not include passwords, authentication tokens, one-time passcodes, or full payment-card details. [incidents]

**eval-bank-chargeback**: How many days will my external bank take to resolve a chargeback for a duplicate payment?
- baseline — abstained (model reported insufficient evidence): The approved material is insufficient and should be escalated for manual review.
- hybrid — answered (citations match retrieved evidence): The purchase is refundable within 30 calendar days when the customer provides the order ID. Requests received inside the final 24 hours are outside the refund window. [refunds]

**eval-shipping-compensation**: What compensation amount is guaranteed if express shipping arrives late?
- baseline — rejected (answer has no citation): The context states that a delivery address can be changed only before the warehouse packs the order. So, if the delivery address is changed before the warehouse packs the order, the customer may not receive the refund.
- hybrid — answered (citations match retrieved evidence): Express shipping usually takes three to five business days. Express shipping usually takes one to two business days. [shipping]

**eval-cafeteria**: Where can I find next week's cafeteria menu?
- baseline — abstained (retrieval score below MIN_SCORE): -
- hybrid — abstained (retrieval score below MIN_SCORE): -

**eval-bicycle**: Can I reserve a bicycle parking space near the office?
- baseline — abstained (retrieval score below MIN_SCORE): -
- hybrid — abstained (retrieval score below MIN_SCORE): -
