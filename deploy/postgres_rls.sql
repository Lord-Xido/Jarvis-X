-- Defense-in-depth tenant isolation for PostgreSQL.
-- The application sets app.tenant_id for each tenant transaction.

DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'memberships','clients','leads','engagements','time_entries','expenses',
        'subscriptions','usage_events','invoices','invoice_lines','payments',
        'employees','leave_requests','vendors','purchase_orders','audit_events'
    ]
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format(
            'CREATE POLICY tenant_isolation ON %I USING '
            || '(tenant_id::text = current_setting(''app.tenant_id'', true)) '
            || 'WITH CHECK (tenant_id::text = current_setting(''app.tenant_id'', true))',
            table_name
        );
    END LOOP;
END $$;
