# Dr Moagi Software Consultancy Cloud Network SaaS Enterprise

## 1. Enterprise system state

The operational company state is

\[
\Sigma_t=(O_t,U_t,C_t,L_t,E_t,T_t,X_t,B_t,I_t,P_t,H_t,V_t,A_t,\Omega_t,\Lambda_t)
\]

where:

- \(O\): tenant organisations and legal entities;
- \(U\): users, employees, memberships, and RBAC;
- \(C\): clients, contacts, and CRM leads;
- \(L\): pipeline and opportunity state;
- \(E\): consultancy engagements;
- \(T\): approved and unapproved time;
- \(X\): expenses and reimbursements;
- \(B\): plans, subscriptions, metered usage, and billing policy;
- \(I\): invoices, invoice lines, balances, and payments;
- \(P\): procurement, vendors, and purchase orders;
- \(H\): employees and leave administration;
- \(V\): enterprise geometric health vectors;
- \(A\): immutable audit events;
- \(\Omega\): retained operational and corrective memory;
- \(\Lambda\): tenant, financial, role, and consistency constraints.

Each accepted command performs

\[
\boxed{\Sigma_{t+1}=\Pi_{\Lambda_t}\left[\mathcal F_I(\Sigma_t)+\Delta\Omega_t\right]}
\]

and any rejected transaction rolls back to \(\Sigma_t\).

## 2. Multi-tenant geometric model

Every engagement is mapped into a three-dimensional enterprise manifold:

\[
G_e=(D_e,F_e,G_e)
\]

with delivery, finance, and governance axes:

\[
D_e=0.45p+0.30q+0.25s
\]

\[
F_e=0.40(1-b)+0.35c+0.25m
\]

\[
G_e=0.50g+0.25a+0.25(1-r)
\]

where \(p\) is progress, \(q\) quality, \(s\) schedule performance, \(b\) budget
utilisation, \(c\) collections, \(m\) margin, \(g\) governance, \(a\) audit
completeness, and \(r\) stated risk.

Health is a weighted geometric mean:

\[
H_e=D_e^{0.40}F_e^{0.35}G_e^{0.25}
\]

and risk is

\[
R_e=\operatorname{clip}(1-H_e+0.25r,0,1).
\]

The enterprise portfolio is the budget-weighted centroid

\[
\bar G=\frac{\sum_e w_eG_e}{\sum_e w_e}
\]

with dispersion

\[
\delta_G=\sqrt{\frac{\sum_e w_e\|G_e-\bar G\|^2}{\sum_e w_e}}.
\]

This state can be passed directly into the Jarvis-X 3D abstraction ANN core for
risk classification, prioritisation, forecasting, and advisory workflows.

## 3. Dual revenue architecture

### 3.1 Consultancy revenue

For approved time entries \(j\), reimbursable expenses \(k\), and retainers \(r\):

\[
B_{consulting}=\sum_j \frac{m_j}{60}h_j+\sum_kx_k+\sum_rr_r.
\]

### 3.2 SaaS platform revenue

For plan fee \(f\), seats \(s\), included seats \(s_0\), extra-seat price \(p_s\),
usage \(u_m\), included usage \(u_{m0}\), and metric price \(p_m\):

\[
B_{platform}=f+p_s\max(0,s-s_0)+\sum_mp_m\max(0,u_m-u_{m0}).
\]

### 3.3 Tax and settlement

\[
Subtotal=\sum_i Line_i
\]

\[
Tax=\operatorname{round}(Subtotal\times rate_{tax})
\]

\[
Total=Subtotal+Tax,
\qquad Balance_{t+1}=\max(0,Balance_t-Payment_t).
\]

Tax rates are configuration values and must be validated by the enterprise's
accounting and tax professionals before production use.

## 4. Corporate administration ecosystem

The operational data model covers:

- legal tenants and business profiles;
- users, memberships, roles, and signed access tokens;
- CRM clients and weighted sales leads;
- engagements, progress, risk, quality, governance, and budgets;
- time sheets and rate cards;
- expenses and reimbursements;
- subscription plans, seats, usage and idempotent usage events;
- consultancy and platform invoices;
- partial and full payment settlement;
- employees, departments, compensation metadata, and leave requests;
- vendors and purchase orders;
- tenant-scoped immutable audit events.

## 5. Security model

Application isolation requires every operation to carry `tenant_id`. PostgreSQL
row-level security is supplied as a second enforcement layer using
`app.tenant_id`; production database sessions must set this value before issuing
tenant-scoped statements.

Roles are ordered as:

```text
viewer < consultant < auditor < operations_manager < finance_admin
       < tenant_owner < platform_admin
```

Passwords use `scrypt`; access tokens use HMAC-SHA256 and contain tenant, user,
role, issue-time, and expiry claims. Production deployments must source secrets
from a cloud secret manager.

## 6. Billing provider architecture

The internal invoice and payment ledger is authoritative. External payment
providers are adapters. The included Stripe adapter supports subscription
Checkout creation and signed webhook verification without making the Stripe SDK
a mandatory dependency.

Provisioning must follow verified subscription/invoice state transitions rather
than trusting browser redirects. Usage events require tenant-scoped idempotency
keys.

## 7. API surface

Core routes:

```text
POST /v1/bootstrap
POST /v1/auth/token
POST /v1/users
POST /v1/clients
POST /v1/leads
POST /v1/engagements
POST /v1/time-entries
POST /v1/expenses
POST /v1/billing/plans
POST /v1/billing/subscriptions
POST /v1/billing/usage
POST /v1/invoices/consultancy
POST /v1/invoices/platform
POST /v1/payments
POST /v1/corporate/employees
POST /v1/corporate/leave
POST /v1/vendors
POST /v1/purchase-orders
GET  /v1/dashboard
POST /v1/billing/stripe/checkout
POST /v1/billing/stripe/webhook
```

## 8. Local deployment

```bash
export DM_TOKEN_SECRET='replace-with-at-least-32-random-characters'
export DM_BOOTSTRAP_TOKEN='replace-bootstrap-token'
docker compose -f docker-compose.saas.yml up --build
```

Bootstrap through the CLI:

```bash
drmoagi-saas bootstrap \
  --company 'Dr Moagi Software Consultancy' \
  --slug dr-moagi \
  --legal-name 'Dr Moagi Software Consultancy (Pty) Ltd' \
  --admin-name 'Platform Administrator' \
  --admin-email admin@example.com \
  --password 'replace-with-a-strong-password' \
  --tax-rate-bps 0
```

## 9. Cloud Run deployment

The repository includes `.github/workflows/deploy-cloud-run.yml`. It builds the
SaaS container, pushes it to Artifact Registry, and deploys to Cloud Run using
Workload Identity Federation. Required repository variables and secrets:

```text
GCP_PROJECT_ID
GCP_REGION
GCP_WORKLOAD_IDENTITY_PROVIDER
GCP_DEPLOY_SERVICE_ACCOUNT
GCP_RUNTIME_SERVICE_ACCOUNT
```

Required Secret Manager entries:

```text
dm-database-url
dm-token-secret
dm-bootstrap-token
stripe-secret-key
stripe-webhook-secret
```

The workflow intentionally does not make the service public. Access should be
provided through authenticated Cloud Run invocation, an API gateway, or an
identity-aware proxy.

## 10. Production gates

Before handling real customers or money:

1. provision managed PostgreSQL with backups and point-in-time recovery;
2. apply `deploy/postgres_rls.sql` using a migration role and a distinct runtime role;
3. configure a verified domain, TLS, WAF/rate limits, and monitoring;
4. configure payment webhooks and replay protection;
5. validate VAT/tax, invoicing, retention, privacy, employment, and accounting
   requirements with qualified South African professionals;
6. add tested backup restoration, disaster recovery, incident response, and
   segregation-of-duties procedures;
7. conduct independent application and infrastructure security reviews.
