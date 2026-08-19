package ru.brombin.ragview;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import ru.brombin.ragview.eval.FnsFaqHttpClient;
import ru.brombin.ragview.eval.FnsFaqSourceConfig;

import java.net.http.HttpRequest;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.Flow;

import static org.assertj.core.api.Assertions.assertThat;

class FnsFaqHttpClientContractTest {

    @Test
    void buildRequest_shouldMatchFnsAjaxContract() throws Exception {
        HttpRequest request = FnsFaqHttpClient.buildRequest(config(), 911, 7, Duration.ofSeconds(30));

        assertThat(request.method()).isEqualTo("POST");
        assertThat(request.uri().toString()).isEqualTo("https://www.nalog.gov.ru/Ajax.html");
        assertThat(request.headers().firstValue("X-Requested-With")).contains("XMLHttpRequest");
        assertThat(request.headers().firstValue("User-Agent").orElseThrow())
                .startsWith("ragview-fns-eval-collector/1.0");
        assertThat(request.headers().firstValue("Referer"))
                .contains("https://www.nalog.gov.ru/rn77/service/kb/?t1=911");
        assertThat(readBody(request))
                .isEqualTo("type=KbSearch&t1=911&t2=&t3=&r=446&cont=&ch=&pgid=7");
    }

    @Test
    void jackson_shouldDecodeCapitalizedAnswerField() throws Exception {
        String raw = "{\"Result\":\"\",\"Success\":true,\"WebIndex\":\"11\",\"Answer\":\"<table></table>\"}";

        Object value = invokeDecode(raw.getBytes(StandardCharsets.UTF_8));

        assertThat(value.toString()).contains("success=true", "answer=<table></table>");
    }

    private static Object invokeDecode(byte[] raw) throws Exception {
        var method = FnsFaqHttpClient.class.getDeclaredMethod("decode", byte[].class, ObjectMapper.class);
        method.setAccessible(true);
        return method.invoke(null, raw, new ObjectMapper());
    }

    private static String readBody(HttpRequest request) throws InterruptedException {
        BodySubscriber subscriber = new BodySubscriber();
        request.bodyPublisher().orElseThrow().subscribe(subscriber);
        return new String(subscriber.bytes(), StandardCharsets.UTF_8);
    }

    private static FnsFaqSourceConfig config() {
        return new FnsFaqSourceConfig(
                1,
                "https://www.nalog.gov.ru/rn77/service/kb/",
                "https://www.nalog.gov.ru/Ajax.html",
                446,
                "seed",
                8,
                500,
                new FnsFaqSourceConfig.Selection(
                        "question", 1, "balanced_by_category",
                        "sha256_rank_without_replacement", "split-seed"),
                new FnsFaqSourceConfig.NearDuplicates("normalized_levenshtein", 0.75),
                new FnsFaqSourceConfig.Scope(
                        "federal", "2026-07-11",
                        "366c21d0599f8742d16584ae2870544de5f71ff652cdea85a54bc76e93ae6d52"),
                List.of(new FnsFaqSourceConfig.Category(911, "НПД")));
    }

    private static final class BodySubscriber implements Flow.Subscriber<ByteBuffer> {
        private final java.io.ByteArrayOutputStream output = new java.io.ByteArrayOutputStream();

        @Override
        public void onSubscribe(Flow.Subscription subscription) {
            subscription.request(Long.MAX_VALUE);
        }

        @Override
        public void onNext(ByteBuffer item) {
            byte[] bytes = new byte[item.remaining()];
            item.get(bytes);
            output.writeBytes(bytes);
        }

        @Override
        public void onError(Throwable throwable) {
            throw new AssertionError(throwable);
        }

        @Override
        public void onComplete() {
        }

        byte[] bytes() {
            return output.toByteArray();
        }
    }
}
