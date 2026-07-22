package com.moagi.omega;

import com.moagi.omega.api.Capability;
import com.moagi.omega.core.BrowserKernel;
import com.moagi.omega.core.CapabilityBroker;
import com.moagi.omega.core.EngineSelector;
import com.moagi.omega.engine.mock.MockBrowserEngine;
import com.moagi.omega.ui.OmegaBrowserFrame;

import javax.swing.*;
import java.nio.file.Path;

public final class Main {
    private Main() {}

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception ignored) {}

            EngineSelector selector = new EngineSelector();
            selector.register(new MockBrowserEngine());

            CapabilityBroker broker = new CapabilityBroker((session, origin, capability, rationale) -> {
                int choice = JOptionPane.showConfirmDialog(
                        null,
                        "Origin: " + origin.serialize()
                                + "\nCapability: " + capability
                                + "\nReason: " + rationale,
                        "Capability request",
                        JOptionPane.YES_NO_OPTION,
                        capability == Capability.CAMERA || capability == Capability.MICROPHONE
                                ? JOptionPane.WARNING_MESSAGE
                                : JOptionPane.QUESTION_MESSAGE
                );
                return choice == JOptionPane.YES_OPTION
                        ? CapabilityBroker.Decision.ALLOW
                        : CapabilityBroker.Decision.DENY;
            });

            Path profile = Path.of(System.getProperty("user.home"), ".moagi-omega-browser-v2");
            BrowserKernel kernel = new BrowserKernel(selector, broker, profile);
            new OmegaBrowserFrame(kernel).setVisible(true);
        });
    }
}
