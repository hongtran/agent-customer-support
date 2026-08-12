"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { clearSession, getMe, getToken, Me, UnauthorizedError } from "@/lib/api";

type State = { status: "loading" } | { status: "ready"; me: Me };

/**
 * Client-side session guard: bounces to /login when there's no usable token.
 *
 * It calls /auth/me rather than trusting the presence of a token in localStorage —
 * a token that has expired or belongs to a deleted customer looks identical in
 * storage, and finding out at the first chat request means the user types a whole
 * message before being kicked out.
 */
export function useSession(): State {
  const router = useRouter();
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    getMe()
      .then((me) => setState({ status: "ready", me }))
      .catch((err) => {
        if (!(err instanceof UnauthorizedError)) clearSession();
        router.replace("/login");
      });
  }, [router]);

  return state;
}

export function logout(router: ReturnType<typeof useRouter>) {
  clearSession();
  router.replace("/login");
}
