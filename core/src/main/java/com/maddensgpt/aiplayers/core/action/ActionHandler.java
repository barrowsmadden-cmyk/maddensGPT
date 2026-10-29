package com.maddensgpt.aiplayers.core.action;

/**
 * Carries out the game actions in {@link ActionCatalog#gameActions()}. Implemented by the
 * Minecraft side of the mod; all calls happen on the server thread.
 */
public interface ActionHandler {
    /**
     * Starts the named action.
     *
     * @param name one of the {@link ActionCatalog} game action names
     * @param args the model's arguments
     * @return the running action; may finish on its first {@link RunningAction#tick()}
     * @throws InvalidArgsException when the arguments are missing, malformed or refer to something
     *                              that does not exist (the message goes back to the model)
     */
    RunningAction start(String name, ActionArgs args) throws InvalidArgsException;
}
