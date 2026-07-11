#ifndef MOAGI_SERVO_HOST_H
#define MOAGI_SERVO_HOST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MOAGI_SERVO_ABI_VERSION 1u

typedef struct moagi_servo_host moagi_servo_host;
typedef struct moagi_servo_session moagi_servo_session;

typedef struct {
    uint32_t abi_version;
    uint32_t viewport_width;
    uint32_t viewport_height;
    uint8_t private_mode;
    const char *profile_partition_utf8;
} moagi_servo_session_config;

typedef void (*moagi_servo_event_callback)(
        const uint8_t *message,
        size_t message_length,
        void *user_data);

moagi_servo_host *moagi_servo_host_create(
        moagi_servo_event_callback callback,
        void *user_data);

void moagi_servo_host_destroy(moagi_servo_host *host);

moagi_servo_session *moagi_servo_session_create(
        moagi_servo_host *host,
        const moagi_servo_session_config *config);

int32_t moagi_servo_session_navigate(
        moagi_servo_session *session,
        const char *uri_utf8);

int32_t moagi_servo_session_stop(moagi_servo_session *session);
void moagi_servo_session_destroy(moagi_servo_session *session);

#ifdef __cplusplus
}
#endif

#endif
