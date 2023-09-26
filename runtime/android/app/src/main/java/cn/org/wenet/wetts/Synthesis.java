package cn.org.wenet.wetts;

public class Synthesis {

    static {
        System.loadLibrary("wetts");
    }

    public static native void init(String modelDir);
    public static native void run(String text, String speaker);
}
