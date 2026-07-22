package com.moagi.omega.ui;

import com.moagi.omega.api.*;
import com.moagi.omega.api.BrowserEvent.*;
import com.moagi.omega.api.Engine.Event;
import com.moagi.omega.api.SemanticScene.Action;
import com.moagi.omega.api.SemanticScene.Node;
import com.moagi.omega.api.SemanticScene.Snapshot;
import com.moagi.omega.api.Surface.CpuFrame;
import com.moagi.omega.api.Surface.Frame;
import com.moagi.omega.core.BrowserKernel;
import com.moagi.omega.core.TransactionJournal.TransactionSnapshot;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.net.URI;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Flow;

/** Swing control surface for the engine-independent browser kernel. */
public final class OmegaBrowserFrame extends JFrame {
    private final BrowserKernel kernel;
    private final JTextField address = new JTextField("https://example.com/");
    private final JLabel status = new JLabel("Kernel ready");
    private final JLabel engine = new JLabel("AUTO");
    private final JCheckBox spatial = new JCheckBox("Spatial scene", true);
    private final SurfacePanel surfacePanel = new SurfacePanel();
    private final DefaultListModel<Node> nodeModel = new DefaultListModel<>();
    private final JList<Node> nodes = new JList<>(nodeModel);
    private final DefaultTableModel transactions = new DefaultTableModel(
            new Object[]{"State", "Command", "Detail", "Revision"}, 0
    ) {
        @Override public boolean isCellEditable(int row, int column) { return false; }
    };

    private UUID sessionId;
    private Snapshot latestSnapshot;

    public OmegaBrowserFrame(BrowserKernel kernel) {
        super("Moagi Ω Browser — Kernel Alpha");
        this.kernel = kernel;
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        setMinimumSize(new Dimension(1080, 720));
        setSize(1450, 900);
        setLocationRelativeTo(null);
        setContentPane(buildUi());
        subscribe();

        addWindowListener(new java.awt.event.WindowAdapter() {
            @Override public void windowClosed(java.awt.event.WindowEvent event) {
                if (sessionId != null) kernel.closeSession(sessionId);
                kernel.close();
            }
        });

        sessionId = kernel.openSession(false, 1120, 700, URI.create(address.getText()));
    }

    private JComponent buildUi() {
        JPanel root = new JPanel(new BorderLayout(8, 8));
        root.setBorder(new EmptyBorder(8, 8, 8, 8));
        root.setBackground(new Color(20, 24, 31));

        JButton navigate = button("Navigate", this::navigate);
        JButton reload = button("Reload", () -> dispatch(new BrowserCommand.Reload()));
        JButton stop = button("Stop", () -> dispatch(new BrowserCommand.Stop()));
        JButton newSession = button("New private session", this::newPrivateSession);
        JButton capability = button("Request geolocation",
                () -> dispatch(new BrowserCommand.RequestCapability(Capability.GEOLOCATION, true)));

        address.addActionListener(event -> navigate());
        spatial.addActionListener(event -> {
            surfacePanel.setSpatial(spatial.isSelected());
            dispatch(new BrowserCommand.SetSpatialMode(spatial.isSelected()));
        });

        JPanel toolbar = new JPanel(new BorderLayout(8, 0));
        toolbar.setBackground(new Color(31, 37, 47));
        JPanel left = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 5));
        left.setOpaque(false);
        left.add(reload);
        left.add(stop);
        left.add(newSession);
        toolbar.add(left, BorderLayout.WEST);
        toolbar.add(address, BorderLayout.CENTER);
        JPanel right = new JPanel(new FlowLayout(FlowLayout.RIGHT, 6, 5));
        right.setOpaque(false);
        right.add(navigate);
        right.add(capability);
        right.add(spatial);
        toolbar.add(right, BorderLayout.EAST);
        root.add(toolbar, BorderLayout.NORTH);

        nodes.setCellRenderer(new NodeRenderer());
        nodes.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        nodes.addMouseListener(new MouseAdapter() {
            @Override public void mouseClicked(MouseEvent event) {
                if (event.getClickCount() == 2) executeSelectedNode();
            }
        });

        JButton executeNode = button("Execute selected action", this::executeSelectedNode);
        JPanel inspector = new JPanel(new BorderLayout(5, 5));
        inspector.setBorder(BorderFactory.createTitledBorder("Normalized semantic scene"));
        inspector.add(new JScrollPane(nodes), BorderLayout.CENTER);
        inspector.add(executeNode, BorderLayout.SOUTH);
        inspector.setPreferredSize(new Dimension(340, 500));

        JSplitPane horizontal = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, surfacePanel, inspector);
        horizontal.setResizeWeight(0.78);
        horizontal.setDividerLocation(1000);

        JTable table = new JTable(transactions);
        table.setAutoCreateRowSorter(true);
        JScrollPane journal = new JScrollPane(table);
        journal.setBorder(BorderFactory.createTitledBorder("Transactional kernel journal"));
        journal.setPreferredSize(new Dimension(1000, 210));

        JSplitPane vertical = new JSplitPane(JSplitPane.VERTICAL_SPLIT, horizontal, journal);
        vertical.setResizeWeight(0.76);
        vertical.setDividerLocation(620);
        root.add(vertical, BorderLayout.CENTER);

        JPanel footer = new JPanel(new BorderLayout());
        footer.setBorder(new EmptyBorder(4, 6, 4, 6));
        footer.add(status, BorderLayout.CENTER);
        footer.add(engine, BorderLayout.EAST);
        root.add(footer, BorderLayout.SOUTH);
        return root;
    }

    private void subscribe() {
        kernel.events().subscribe(new Flow.Subscriber<>() {
            @Override public void onSubscribe(Flow.Subscription subscription) {
                subscription.request(Long.MAX_VALUE);
            }
            @Override public void onNext(BrowserEvent item) {
                SwingUtilities.invokeLater(() -> handle(item));
            }
            @Override public void onError(Throwable throwable) {
                SwingUtilities.invokeLater(() -> status.setText("Kernel stream error: " + throwable));
            }
            @Override public void onComplete() {}
        });
    }

    private void handle(BrowserEvent event) {
        if (event instanceof SessionChanged changed) {
            if (sessionId == null || changed.sessionId().equals(sessionId)) {
                engine.setText(changed.engineId() + " / " + changed.state());
            }
            return;
        }
        if (sessionId == null) return;

        if (event instanceof TransactionChanged changed && changed.sessionId().equals(sessionId)) {
            addTransaction(changed.transaction());
        } else if (event instanceof CapabilityChanged changed && changed.sessionId().equals(sessionId)) {
            status.setText(changed.decision());
        } else if (event instanceof EngineEvent engineEvent && engineEvent.sessionId().equals(sessionId)) {
            Event payload = engineEvent.event();
            if (payload instanceof Event.Correlated correlated) {
                payload = correlated.event();
            }
            if (payload instanceof Event.Status message) {
                status.setText(message.message());
            } else if (payload instanceof Event.NavigationStarted started) {
                status.setText("Navigating: " + started.uri());
            } else if (payload instanceof Event.NavigationCommitted committed) {
                address.setText(committed.uri().toString());
                status.setText("Committed: " + committed.title());
            } else if (payload instanceof Event.NavigationFailed failed) {
                status.setText("Failed: " + failed.reason());
            } else if (payload instanceof Event.FrameReady ready) {
                surfacePanel.setFrame(ready.frame());
            } else if (payload instanceof Event.SnapshotReady ready) {
                updateSnapshot(ready.snapshot());
            } else if (payload instanceof Event.Crashed crashed) {
                status.setText("Engine crashed: " + crashed.reason());
            }
        }
    }

    private void updateSnapshot(Snapshot snapshot) {
        latestSnapshot = snapshot;
        nodeModel.clear();
        snapshot.nodes().stream()
                .filter(node -> node.id() != 0)
                .forEach(nodeModel::addElement);
        surfacePanel.setSnapshot(snapshot);
    }

    private void addTransaction(TransactionSnapshot tx) {
        transactions.addRow(new Object[]{
                tx.state(), tx.command(), tx.detail(),
                tx.committedRevision() < 0 ? "—" : tx.committedRevision()
        });
        int last = transactions.getRowCount() - 1;
        if (last >= 0) transactions.fireTableRowsUpdated(last, last);
    }

    private void navigate() {
        try {
            String raw = address.getText().strip();
            URI uri = raw.matches("^[a-zA-Z][a-zA-Z0-9+.-]*:.*")
                    ? URI.create(raw)
                    : URI.create("https://" + raw);
            dispatch(new BrowserCommand.Navigate(uri));
        } catch (RuntimeException ex) {
            status.setText("Invalid URI: " + ex.getMessage());
        }
    }

    private void newPrivateSession() {
        if (sessionId != null) kernel.closeSession(sessionId);
        transactions.setRowCount(0);
        nodeModel.clear();
        sessionId = kernel.openSession(true, 1120, 700, URI.create("mock://private/start"));
        address.setText("mock://private/start");
        status.setText("Private session created");
    }

    private void executeSelectedNode() {
        Node selected = nodes.getSelectedValue();
        if (selected == null) return;
        Action action = selected.actions().contains(Action.CLICK)
                ? Action.CLICK
                : selected.actions().stream().findFirst().orElse(Action.FOCUS);
        dispatch(new BrowserCommand.SemanticAction(selected.id(), action, Map.of()));
    }

    private void dispatch(BrowserCommand command) {
        if (sessionId == null) return;
        kernel.dispatch(sessionId, command);
    }

    private static JButton button(String text, Runnable action) {
        JButton button = new JButton(text);
        button.addActionListener(event -> action.run());
        return button;
    }

    private static final class NodeRenderer extends DefaultListCellRenderer {
        @Override
        public Component getListCellRendererComponent(
                JList<?> list, Object value, int index, boolean selected, boolean focused
        ) {
            JLabel label = (JLabel) super.getListCellRendererComponent(
                    list, value, index, selected, focused);
            if (value instanceof Node node) {
                label.setText("<html><b>" + node.role() + "</b> — "
                        + node.accessibleName() + "<br><small>id=" + node.id()
                        + " z=" + node.layerDepth() + " actions=" + node.actions()
                        + "</small></html>");
                label.setBorder(new EmptyBorder(6, 6, 6, 6));
            }
            return label;
        }
    }

    private static final class SurfacePanel extends JPanel {
        private BufferedImage image;
        private Snapshot snapshot;
        private boolean spatial = true;

        SurfacePanel() {
            setBackground(new Color(8, 11, 16));
            setBorder(BorderFactory.createTitledBorder("Engine frame surface"));
        }

        void setFrame(Frame frame) {
            if (frame instanceof CpuFrame cpu) {
                BufferedImage next = new BufferedImage(cpu.width(), cpu.height(), BufferedImage.TYPE_INT_ARGB);
                next.setRGB(0, 0, cpu.width(), cpu.height(), cpu.argb(), 0, cpu.width());
                image = next;
                repaint();
            }
        }

        void setSnapshot(Snapshot snapshot) {
            this.snapshot = snapshot;
            repaint();
        }

        void setSpatial(boolean spatial) {
            this.spatial = spatial;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics graphics) {
            super.paintComponent(graphics);
            Graphics2D g = (Graphics2D) graphics.create();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int pad = 24;
            int availableW = Math.max(1, getWidth() - pad * 2);
            int availableH = Math.max(1, getHeight() - pad * 2);
            if (image != null) {
                double scale = Math.min(availableW / (double) image.getWidth(),
                        availableH / (double) image.getHeight());
                int w = (int) Math.round(image.getWidth() * scale);
                int h = (int) Math.round(image.getHeight() * scale);
                int x = (getWidth() - w) / 2;
                int y = (getHeight() - h) / 2;
                g.drawImage(image, x, y, w, h, null);

                if (spatial && snapshot != null) {
                    double sx = w / 1200.0;
                    double sy = h / 760.0;
                    for (Node node : snapshot.nodes()) {
                        if (node.id() == 0) continue;
                        int depth = node.layerDepth();
                        var b = node.bounds();
                        int bx = x + (int) Math.round((b.x() + depth * 3) * sx);
                        int by = y + (int) Math.round((b.y() - depth * 3) * sy);
                        int bw = (int) Math.round(b.width() * sx);
                        int bh = (int) Math.round(b.height() * sy);
                        g.setColor(new Color(130, 210, 255, 120));
                        g.setStroke(new BasicStroke(1.2f));
                        g.drawRect(bx, by, bw, bh);
                        g.setColor(new Color(220, 245, 255));
                        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 10));
                        g.drawString("z=" + depth, bx + 4, by + 12);
                    }
                }
            } else {
                g.setColor(new Color(170, 185, 205));
                g.drawString("Awaiting engine frame…", pad, pad + 20);
            }
            g.dispose();
        }
    }
}
