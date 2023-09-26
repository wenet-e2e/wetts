package cn.org.wenet.wetts;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final String LOG_TAG = "WETTS";
    private static final List<String> resource = Arrays.asList("frontend", "vits");

    private void copyFile(String filename) throws Exception {
        File dst = new File(this.getFilesDir(), filename);
        Log.i(LOG_TAG, "Copying " + filename + " to " + dst.getAbsolutePath());
        InputStream is = this.getAssets().open(filename);
        OutputStream os = new FileOutputStream(dst);
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
            os.write(buffer, 0, read);
        }
        os.flush();
    }

    public void copyFileOrDir(String path) throws Exception {
        AssetManager assetMgr = this.getAssets();
        String assets[] = assetMgr.list(path);
        if (assets.length == 0) {
            copyFile(path);
        } else {
            File dir = new File(this.getFilesDir(), path);
            if (!dir.exists()) {
                dir.mkdir();
            }
            for (int i = 0; i < assets.length; ++i) {
                copyFileOrDir(path.length() == 0 ? assets[i] : path + "/" + assets[i]);
            }
        }
    }

    public void assetsInit() throws Exception {
        // Unzip all files in resource from assets to context.
        // Note: Uninstall the APP will remove the resource files in the context.
        for (String dir : resource) {
            copyFileOrDir(dir);
        }
    }

    public String[] speakersInit(String speaker_table) throws Exception {
        List<String> speakers = new LinkedList<>();
        BufferedReader in = new BufferedReader(new FileReader(speaker_table));
        String speaker;
        while ((speaker = in.readLine()) != null) {
            speakers.add(speaker.split(" ")[0]);
        }
        return speakers.toArray(new String[0]);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            assetsInit();
        } catch (Exception e) {
            Log.e(LOG_TAG, "Error process asset files to file path");
        }
        Synthesis.init(getFilesDir().getPath());

        String speakers[] = new String[0];
        try {
            speakers = speakersInit(getFilesDir().getPath() + "/vits/speaker.txt");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        Spinner spinner = (Spinner) findViewById(R.id.spinner);
        ArrayAdapter<String> adapter = new ArrayAdapter<>
                (this, android.R.layout.simple_expandable_list_item_1, speakers);
        spinner.setAdapter(adapter);

        EditText editText = findViewById(R.id.editText);
        Button button = findViewById(R.id.button);
        button.setText("Start Synthesis");
        button.setOnClickListener(view -> {
            MediaPlayer player = new MediaPlayer();
            try {
                String text = editText.getText().toString();
                String speaker = spinner.getSelectedItem().toString();
                Log.e(LOG_TAG, text);
                Log.e(LOG_TAG, speaker);
                Synthesis.run(text, speaker);
                player.setDataSource(this.getFilesDir() + "/audio.wav");
                player.prepare();
                player.start();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }
}
